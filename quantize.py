import torch
from torch.quantization import (
    quantize_dynamic,
    prepare,
    convert,
    get_default_qconfig,
    fuse_modules,
)
from torch.quantization.observer import MinMaxObserver, HistogramObserver

from models.mobilenetv3_lraspp import lraspp_mobilenet_v3_large
from dataset import get_dataloader
from trainer import MODEL_STATE_DICT_NAME

import time
import numpy as np
from scipy.spatial.distance import cosine
import cv2
import matplotlib.pyplot as plt

import os
import gc
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from memory_profiler import memory_usage


def _fuse_conv_bn_act(module, is_qat):
    if isinstance(module, torch.nn.Sequential):
        # Case 1: Conv + BN + Hardswish
        if (
            len(module) == 3
            and isinstance(module[0], torch.nn.Conv2d)
            and isinstance(module[1], torch.nn.BatchNorm2d)
            and isinstance(module[2], torch.nn.Hardswish)
        ):
            # Manually fuse Conv+BN
            fused_conv = torch.quantization.fuse_conv_bn(is_qat, module[0], module[1])
            # Replace original layers with fused Conv
            module[0] = fused_conv
            module[1] = torch.nn.Identity()  # BN becomes no-op
            module[2] = torch.nn.Identity()  # Hardswish remains separate (unfused)

        # Case 2: Conv + BN + ReLU (standard fusion)
        elif (
            len(module) == 3
            and isinstance(module[0], torch.nn.Conv2d)
            and isinstance(module[1], torch.nn.BatchNorm2d)
            and isinstance(module[2], torch.nn.ReLU)
        ):
            fuse_modules(module, [["0", "1", "2"]], inplace=True)

        # Case 3: Conv + BN (no activation)
        elif (
            len(module) == 2
            and isinstance(module[0], torch.nn.Conv2d)
            and isinstance(module[1], torch.nn.BatchNorm2d)
        ):
            fuse_modules(module, [["0", "1"]], inplace=True)

    return module


def _fuse_inverted_residual(module, is_qat):
    if hasattr(module, "block") and isinstance(module.block, torch.nn.Sequential):
        for submodule in module.block:
            if isinstance(submodule, torch.nn.Sequential):
                _fuse_conv_bn_act(submodule, is_qat)
    return module


def _fuse_backbone(backbone, is_qat):
    for name, module in backbone.named_children():
        if isinstance(module, torch.nn.Sequential):
            _fuse_conv_bn_act(module, is_qat)
        elif hasattr(module, "block"):
            _fuse_inverted_residual(module, is_qat)
    return backbone


def _fuse_model(model, is_qat=False):
    # Fuse backbone
    model.backbone = _fuse_backbone(model.backbone, is_qat)

    # Fuse classifier (if exists)
    if hasattr(model, "classifier"):
        if hasattr(model.classifier, "cbr") and isinstance(
            model.classifier.cbr, torch.nn.Sequential
        ):
            _fuse_conv_bn_act(model.classifier.cbr, is_qat)
    return model


def fp32_to_int8(model_path, calibration_dataloader, dynamic=True, platform="fbgemm"):
    model_fp32 = lraspp_mobilenet_v3_large(num_classes=2)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))[
        MODEL_STATE_DICT_NAME
    ]
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model_fp32.load_state_dict(state_dict, strict=True)
    model_fp32.eval()

    if not dynamic:
        model_fp32 = _fuse_model(model_fp32)

        model_fp32.qconfig = get_default_qconfig(platform)
        model_fp32_prepare = prepare(model_fp32, inplace=False)

        with torch.no_grad():
            for image, _ in calibration_dataloader:
                image = image.to(torch.device("cpu"))
                model_fp32_prepare(image)

        model_int8 = convert(model_fp32_prepare, inplace=False)
    else:
        model_int8 = quantize_dynamic(
            model_fp32,
            qconfig_spec={torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8,
        )
    return model_fp32, model_int8


def performance_check(model_fp32, model_int8, calibration_dataloader):
    model_fp32.eval()
    model_int8.eval()

    with torch.no_grad():
        for image, _ in calibration_dataloader:
            image = image.to(torch.device("cpu"))

            s_time = time.time()
            output_fp32 = model_fp32(image)
            e_time = time.time()
            print(f"FP32 inference time: {e_time - s_time:.4f} seconds")
            s_time = time.time()
            output_int8 = model_int8(image)
            e_time = time.time()
            print(f"INT8 inference time: {e_time - s_time:.4f} seconds")

            fp32_mean = output_fp32.mean().item()
            int8_mean = output_int8.mean().item()
            print(f"FP32 mean: {fp32_mean}, INT8 mean: {int8_mean}")
            print(
                f"accuracy degradation: {abs(fp32_mean - int8_mean) / fp32_mean * 100:.2f}%"
            )


class OnnxDataReader(CalibrationDataReader):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()
        self.iter = iter(self.data)

    def load_data(self):
        data = []
        all_files = sorted(
            [
                f
                for f in os.listdir(self.data_path)
                if os.path.isfile(os.path.join(self.data_path, f))
            ]
        )
        for file in all_files:
            gray = cv2.imread(os.path.join(self.data_path, file), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"Warning: Failed to load {file}")
                continue
            gray = cv2.resize(gray, (640, 480))
            # normalize to [0, 1]
            gray = gray.astype(np.float32) / 255.0
            # and normalize by mu=0.4817, sigma=0.2591
            gray = (gray - 0.4817) / 0.2591
            # add batch dimension
            gray = np.expand_dims(gray, axis=0)
            gray = np.expand_dims(gray, axis=0)
            data.append(gray)
        return data

    def get_next(self):
        try:
            data = next(self.iter)
            return {"input": data}
        except StopIteration:
            return None

    def reset(self):
        self.iter = iter(self.data)


def onnx_quant_fp32_to_int8(model_path, data_path, output_path):
    data_reader = OnnxDataReader(data_path)

    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=data_reader,
        per_channel=False,
        reduce_range=False,
        quant_format=QuantType.QUInt8,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"Quantized model saved to {output_path}")


def benchmark_model_speed(sess, data_reader, num_runs=100):
    # warmup draw
    dummy_input = data_reader.get_next()["input"]
    _ = sess.run(["output"], {"input": dummy_input})

    total_time = 0.0
    for _ in range(num_runs):
        data = data_reader.get_next()
        if data is None:
            data_reader.reset()
            data = data_reader.get_next()

        start_time = time.time()
        outputs = sess.run(["output"], data)
        end_time = time.time()

        total_time += end_time - start_time

    avg_time = (total_time / num_runs) * 1000
    print(f"Average inference time: {avg_time:.2f} ms")
    return avg_time


def benchmark_model_memory(sess, data_reader):
    # warmup draw
    data = data_reader.get_next()

    mem_usage = memory_usage(proc=(sess.run, (["output"], data)), interval=0.1)
    print(f"Peak memory usage: {max(mem_usage):.2f} MB")
    return max(mem_usage)


def compute_similarity(fp32_output, int8_output):
    mse = np.mean((fp32_output - int8_output) ** 2)
    cos_sim = 1 - cosine(fp32_output.flatten(), int8_output.flatten())
    # signal to noise ratio(SNR)
    signal_power = np.mean(fp32_output**2)
    noise_power = np.mean((fp32_output - int8_output) ** 2)
    snr = (
        10 * np.log10(signal_power / noise_power) if noise_power != 0 else float("inf")
    )

    print(f"MSE: {mse:.6f}")
    print(f"Cosine Similarity: {cos_sim:.4f}")
    print(f"SNR: {snr:.2f} dB")
    return mse, cos_sim, snr


def plot_difference(fp32_output, int8_output):
    # Flatten outputs
    fp32_flat = fp32_output.flatten()
    int8_flat = int8_output.flatten()
    
    # Calculate errors
    errors = fp32_flat - int8_flat
    
    plt.figure(figsize=(12, 4))
    
    # Histogram of errors
    plt.subplot(1, 2, 1)
    plt.hist(errors, bins=50)
    plt.title("Error Distribution (FP32 - INT8)")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    
    # Scatter plot (subsampled)
    plt.subplot(1, 2, 2)
    sample_idx = np.random.choice(len(fp32_flat), size=1000, replace=False)
    plt.scatter(fp32_flat[sample_idx], int8_flat[sample_idx], alpha=0.3)
    plt.plot([min(fp32_flat), max(fp32_flat)], [min(fp32_flat), max(fp32_flat)], 'r--')
    plt.xlabel("FP32 Output")
    plt.ylabel("INT8 Output")
    plt.title("Output Correlation")
    
    plt.tight_layout()
    plt.show()



def compare_models(fp32_path, int8_path, data_reader):
    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    print("\n=== Speed Benchmark ===")
    fp32_time = benchmark_model_speed(sess_fp32, data_reader)
    int8_time = benchmark_model_speed(sess_int8, data_reader)
    print(f"\nSpeedup: {fp32_time / int8_time:.2f}x")

    print("\n=== Memory Benchmark ===")
    fp32_memory = benchmark_model_memory(sess_fp32, data_reader)
    int8_memory = benchmark_model_memory(sess_int8, data_reader)
    print(f"\nMemory Reduction: {fp32_memory / int8_memory:.2f}x")

    print("\n=== Output Comparison ===")
    fp32_output = None
    int8_output = None
    for _ in range(10):
        data = data_reader.get_next()
        if data is None:
            data_reader.reset()
            data = data_reader.get_next()

        fp32_output = sess_fp32.run(["output"], data)[0]
        int8_output = sess_int8.run(["output"], data)[0]

        mse, cos_sim, snr = compute_similarity(fp32_output, int8_output)

        print(f"MSE: {mse:.6f}, Cosine Similarity: {cos_sim:.4f}, SNR: {snr:.2f} dB")
    plot_difference(fp32_output, int8_output)

    return {
        "fp32_latency_ms": fp32_time,
        "int8_latency_ms": int8_time,
        "speedup": fp32_time / int8_time,
        "fp32_mem_MiB": fp32_memory,
        "int8_mem_MiB": int8_memory,
        "mem_reduction": fp32_memory / int8_memory,
        "mse": mse,
        "cosine_similarity": cos_sim,
        "snr_db": snr,
    }


def main():
    model_path = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models/lraspp_mobilenet_v3_large/run_20250411_124619/lraspp_mobilenet_v3_large_254_iou_0.9229.pth"
    data_path = "/home/william/extdisk/data/ACE20k/ACE20k_sky"
    _, val_dataloader = get_dataloader(data_path)
    model_fp32, model_int8 = fp32_to_int8(
        model_path, val_dataloader, dynamic=False, platform="fbgemm"
    )
    performance_check(model_fp32, model_int8, val_dataloader)
    gc.collect()


def main_onnx():
    model_path = "onnx/mbv3_1ch_fp32_opsetv17_simp.onnx"
    data_path = "/home/william/extdisk/data/ACE20k/ACE20k_sky/images/validation"
    output_path = "onnx/mbv3_1ch_fp32_opsetv17_simp_int8.onnx"
    onnx_quant_fp32_to_int8(model_path, data_path, output_path)
    gc.collect()


def main_compare():
    fp32_path = "onnx/mbv3_1ch_fp32_opsetv17_simp.onnx"
    int8_path = "onnx/mbv3_1ch_fp32_opsetv17_simp_int8.onnx"
    data_path = "/home/william/extdisk/data/ACE20k/ACE20k_sky/images/validation"
    data_reader = OnnxDataReader(data_path)
    res = compare_models(fp32_path, int8_path, data_reader)
    print(res)
    gc.collect()


if __name__ == "__main__":
    main_compare()
