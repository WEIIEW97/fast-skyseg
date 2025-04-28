from pathlib import Path
import gc

import cv2
import numpy as np
import onnxruntime as ort

from tqdm import tqdm


def run_inference(onnx_session, input_size, image):
    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    x = cv2.resize(image, dsize=(input_size[0], input_size[1]))
    x = x.astype(np.float32) / 255
    mean = 0.4817
    std = 0.2686
    x = (x - mean) / std

    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=0)

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})
    
    # post process
    logits = onnx_result[0]
    pred = np.argmax(logits, axis=1)
    pred = np.squeeze(pred, axis=0)
    binary_mask = (pred == 1).astype(np.uint8)  # Assuming class 1 is the target class
    return binary_mask


if __name__ == "__main__":
    test_dir = Path("/home/william/extdisk/data/motorEV/FC_20250425/Infrared_L_0_calib")
    out_dir = Path("/home/william/extdisk/data/motorEV/FC_20250425/mask_onnx_int8")
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = "onnx/mbv3_1ch_fp32_opsetv17_simp_int8.onnx"
    available_providers = ort.get_available_providers()
    print("Available Execution Providers:", available_providers)
    for image_path in tqdm(test_dir.glob("*.png")):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        # transpose
        image = np.rot90(image, k=1)
        onnx_session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        # Run inference
        mask = run_inference(onnx_session, (640, 480), image)
        mask = mask * 255
        # tranpose back
        mask = np.rot90(mask, k=3)
        cv2.imwrite(out_dir / image_path.name, mask)

    gc.collect()
