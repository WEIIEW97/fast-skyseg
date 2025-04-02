import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image

from models.mobilenetv3_lraspp import lraspp_mobilenet_v3_large
from torchvision import transforms
from tqdm import tqdm


def load_model(model_path, num_classes, device):
    """Load the trained model (handles both DDP and single-GPU saved models)."""
    model = lraspp_mobilenet_v3_large(num_classes=num_classes)

    # Load state dict (handles DDP wrapper if present)
    state_dict = torch.load(model_path, map_location=device)
    if (
        "module." in list(state_dict.keys())[0]
    ):  # Check if model was saved with DDP wrapper
        state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }  # Remove "module." prefix

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return model


def preprocess_image(image_path, fixed_size, device, transpose=False):
    """Preprocess the input image for inference."""
    transform = transforms.Compose(
        [
            transforms.Resize(fixed_size),
            transforms.Grayscale(num_output_channels=1),  # RGB --> Grayscale
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3-channel
        ]
    )
    image = Image.open(image_path).convert("RGB")
    if transpose:
        image = image.transpose(Image.Transpose.ROTATE_90)
        original_size = (image.size[1], image.size[0])  # (height, width)
    else:
        original_size = image.size
    image = transform(image).unsqueeze(0).to(device)  # Add batch dim and move to device
    return image, original_size


def postprocess_output(output, original_size, transpose=False):
    """Convert model output to a segmentation mask."""
    pred = torch.argmax(output, dim=1).squeeze(
        0
    )  # Remove batch dim and convert to numpy
    binary_mask = (pred == 1).long()
    binary_mask_np = binary_mask.cpu().numpy()
    pred = Image.fromarray(binary_mask_np.astype(np.uint8))
    if transpose:
        pred = pred.transpose(Image.Transpose.ROTATE_270)
    pred = pred.resize(
        original_size, Image.NEAREST
    )  # Resize to original image dimensions
    return pred


def inference(model, image_path, fixed_size, device, transpose=False):
    """Perform inference on a single image."""
    image, original_size = preprocess_image(image_path, fixed_size, device, transpose)
    with torch.no_grad():
        output = model(image)
    pred = postprocess_output(output, original_size, transpose)
    return pred


if __name__ == "__main__":
    model_path = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models/lraspp_mobilenet_v3_large/run_20250402_152418/lraspp_mobilenet_v3_large_93_iou_0.9382.pth"
    num_classes = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_size = (480, 640)

    model = load_model(model_path, num_classes, device)
    test_dir = "/home/william/extdisk/data/feynman/FC1/GY2DSH24GV0023/Infrared_L_0_calib/"
    pred_dir = "/home/william/extdisk/data/feynman/FC1/GY2DSH24GV0023/Infrared_L_0_calib/predictions_1ch"
    os.makedirs(pred_dir, exist_ok=True)
    image_paths = [
        f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))
    ]
    for p in tqdm(image_paths):
        image_path = os.path.join(test_dir, p)
        pred = inference(model, image_path, fixed_size, device, True)
        pred = np.array(pred) * 255
        pred = Image.fromarray(pred.astype(np.uint8))
        pred.save(os.path.join(pred_dir, p.replace(".jpg", ".png")))
    print("done!")
