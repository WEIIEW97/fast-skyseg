import torch
from torch.quantization import quantize_dynamic, prepare, convert, get_default_qconfig, fuse_modules
from torch.quantization.observer import MinMaxObserver, HistogramObserver

from models.mobilenetv3_lraspp import lraspp_mobilenet_v3_large
from dataset import get_dataloader
from trainer import MODEL_STATE_DICT_NAME

import time

def _fuse_conv_bn_act(module, is_qat):
    if isinstance(module, torch.nn.Sequential):
        # Case 1: Conv + BN + Hardswish
        if (len(module) == 3 and 
            isinstance(module[0], torch.nn.Conv2d) and 
            isinstance(module[1], torch.nn.BatchNorm2d) and 
            isinstance(module[2], torch.nn.Hardswish)):
            # Manually fuse Conv+BN
            fused_conv = torch.quantization.fuse_conv_bn(is_qat, module[0], module[1])
            # Replace original layers with fused Conv
            module[0] = fused_conv
            module[1] = torch.nn.Identity()  # BN becomes no-op
            module[2] = torch.nn.Identity()  # Hardswish remains separate (unfused)
        
        # Case 2: Conv + BN + ReLU (standard fusion)
        elif (len(module) == 3 and 
              isinstance(module[0], torch.nn.Conv2d) and 
              isinstance(module[1], torch.nn.BatchNorm2d) and 
              isinstance(module[2], torch.nn.ReLU)):
            fuse_modules(module, [["0", "1", "2"]], inplace=True)
        
        # Case 3: Conv + BN (no activation)
        elif (len(module) == 2 and 
              isinstance(module[0], torch.nn.Conv2d) and 
              isinstance(module[1], torch.nn.BatchNorm2d)):
            fuse_modules(module, [["0", "1"]], inplace=True)
    
    return module

def _fuse_inverted_residual(module, is_qat):
    if hasattr(module, 'block') and isinstance(module.block, torch.nn.Sequential):
        for submodule in module.block:
            if isinstance(submodule, torch.nn.Sequential):
                _fuse_conv_bn_act(submodule, is_qat)
    return module

def _fuse_backbone(backbone, is_qat):
    for name, module in backbone.named_children():
        if isinstance(module, torch.nn.Sequential):
            _fuse_conv_bn_act(module, is_qat)
        elif hasattr(module, 'block'):
            _fuse_inverted_residual(module, is_qat)
    return backbone

def _fuse_model(model, is_qat=False):
    # Fuse backbone
    model.backbone = _fuse_backbone(model.backbone, is_qat)
    
    # Fuse classifier (if exists)
    if hasattr(model, 'classifier'):
        if hasattr(model.classifier, 'cbr') and isinstance(model.classifier.cbr, torch.nn.Sequential):
            _fuse_conv_bn_act(model.classifier.cbr, is_qat)
    return model


def fp32_to_int8(model_path, calibration_dataloader, dynamic=True, platform="fbgemm"):
    model_fp32 = lraspp_mobilenet_v3_large(num_classes=2)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))[MODEL_STATE_DICT_NAME]
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

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
            print(f"accuracy degradation: {abs(fp32_mean - int8_mean) / fp32_mean * 100:.2f}%")


def main():
    model_path = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models/lraspp_mobilenet_v3_large/run_20250411_124619/lraspp_mobilenet_v3_large_254_iou_0.9229.pth"
    data_path = "/home/william/extdisk/data/ACE20k/ACE20k_sky"
    _, val_dataloader = get_dataloader(data_path)
    model_fp32, model_int8 = fp32_to_int8(model_path, val_dataloader, dynamic=False, platform='fbgemm')
    performance_check(model_fp32, model_int8, val_dataloader)

if __name__ == "__main__":
    main()

