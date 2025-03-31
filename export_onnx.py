import torch
from models import mobilenetv3_lraspp

def onnx_export(model, input_tensor, output_path):
    """
    Export the model to ONNX format.
    
    Args:
        model (torch.nn.Module): The model to export.
        input_tensor (torch.Tensor): A sample input tensor for tracing.
        output_path (str): The path to save the exported ONNX model.
    """
    torch.onnx.export(
        model,
        input_tensor,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Model exported to {output_path}")
    print("Model exported to ONNX format successfully.")

if __name__ == "__main__":
    # Example usage
    model = mobilenetv3_lraspp.lraspp_mobilenet_v3_large(num_classes=2)
    model.eval()  

    dummy_input = torch.randn(1, 3, 480, 640) 
    output_path = "onnx/lraspp_mobilenetv3.onnx"
    onnx_export(model, dummy_input, output_path)