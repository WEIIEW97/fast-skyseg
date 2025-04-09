import torch
import torch.nn as nn
import onnx
from onnx.tools import update_model_dims
from models.mobilenetv3_lraspp import lraspp_mobilenet_v3_large
from models.fast_scnn import fast_scnn
from models.bisenetv2 import bisenetv2
from models.u2net import u2net
from models import fast_scnn


class ModelWrapper(nn.Module):
    def __init__(self, original_model):
        super(ModelWrapper, self).__init__()
        self.model = original_model

    def forward(self, x):
        return self.model(x)[0]


def onnx_export(model, input_tensor, output_path, wrapper=False, fixed_batch_size=None):
    """
    Export the model to ONNX format.

    Args:
        model (torch.nn.Module): The model to export.
        input_tensor (torch.Tensor): A sample input tensor for tracing.
        output_path (str): The path to save the exported ONNX model.
    """

    if wrapper:
        model_wrapper = ModelWrapper(model)
        model = model_wrapper.eval()

    dynamic_axes = None if fixed_batch_size else {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

    torch.onnx.export(
        model,
        input_tensor,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"Model exported to {output_path}")
    print("Model exported to ONNX format successfully.")


if __name__ == "__main__":
    # Example usage
    num_classes = 2
    # model_path = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models/lraspp_mobilenet_v3_large/run_20250402_152418/lraspp_mobilenet_v3_large_93_iou_0.9382.pth"
    # model_path = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models/fast_scnn/run_20250402_160813/fast_scnn_159_iou_0.9037.pth"
    # model_path = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models/u2net/run_20250402_165359/u2net_418_iou_0.9322.pth"
    model_path = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models/bisenetv2/run_20250402_161803/bisenetv2_299_iou_0.9123.pth"
    # model = lraspp_mobilenet_v3_large(num_classes=2)
    # model = fast_scnn(num_classes=2, aux=True)
    model = bisenetv2(num_classes=num_classes, aux_mode='train')
    # model = u2net(num_classes=num_classes, model_type='full')

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 1, 480, 640)
    output_path = "onnx/bisenetv2_1ch_fixedB.onnx"
    onnx_export(model, dummy_input, output_path, wrapper=True, fixed_batch_size=1)
