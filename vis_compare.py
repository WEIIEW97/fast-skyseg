import torch
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

from models.mobilenetv3_lraspp import lraspp_mobilenet_v3_large, LRASPP
from models.fast_scnn import fast_scnn
from models.bisenetv2 import bisenetv2, BiSeNetV2
from models.u2net import u2net, U2NET
from trainer import MODEL_STATE_DICT_NAME

# Model paths
model_path_mbv3 = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models/lraspp_mobilenet_v3_large/run_20250411_124619/lraspp_mobilenet_v3_large_254_iou_0.9229.pth"
model_path_fast_scnn = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models/fast_scnn/run_20250402_160813/fast_scnn_159_iou_0.9037.pth"
model_path_bisenetv2 = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models/bisenetv2/run_20250411_152202/bisenetv2_349_iou_0.8394.pth"
model_path_u2net_full = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models/u2net/run_20250402_165359/u2net_418_iou_0.9322.pth"
model_path_u2net_lite = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models/u2net/u2net_lite/run_20250411_152516/u2net_417_iou_0.8900.pth"

# Model initialization
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
model_fast_scnn = fast_scnn(num_classes=num_classes, aux=True).to(device)
model_mbv3 = lraspp_mobilenet_v3_large(num_classes=num_classes).to(device)
model_bisenetv2 = bisenetv2(num_classes=num_classes, aux_mode='train').to(device)
model_u2net_full = u2net(num_classes=num_classes, model_type='full').to(device)
model_u2net_lite = u2net(num_classes=num_classes, model_type='lite').to(device)

# Load model weights
def load_model_weights(model, model_path, u2net_type='lite'):
    if isinstance(model, (LRASPP, BiSeNetV2, U2NET)):
        if u2net_type == 'lite':
            state_dict = torch.load(model_path, map_location=device)[MODEL_STATE_DICT_NAME]
        else:
            state_dict = torch.load(model_path, map_location=device)
    else:
        state_dict = torch.load(model_path, map_location=device)
    
    # Handle DataParallel/DistributedDataParallel saved models
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

model_fast_scnn = load_model_weights(model_fast_scnn, model_path_fast_scnn)
model_mbv3 = load_model_weights(model_mbv3, model_path_mbv3)
model_bisenetv2 = load_model_weights(model_bisenetv2, model_path_bisenetv2)
model_u2net_full = load_model_weights(model_u2net_full, model_path_u2net_full, u2net_type='full')
model_u2net_lite = load_model_weights(model_u2net_lite, model_path_u2net_lite, u2net_type='lite')

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
        if isinstance(output, (tuple, list)):
            output = output[0]
    pred = postprocess_output(output, original_size, transpose)
    return pred


def vis_compare(model_factory, input_path, output_path, nrow=2, figsize=(20, 10), fixed_size=(480, 640), device="cuda", transpose=False):
    """Compare model predictions visually"""
    image = np.array(Image.open(input_path).convert("RGB"))
    num_models = len(model_factory)
    ncol = (num_models + 1 + nrow - 1) // nrow  # +1 for original image
    
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = axes.ravel() if isinstance(axes, np.ndarray) else [axes]
    
    # Show original
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=10)
    
    # Process each model
    for i, (name, model) in enumerate(model_factory.items(), start=1):
        try:
            pred = inference(model, input_path, fixed_size, device, transpose)
            pred = np.array(pred) * 255
            axes[i].set_title(f"{name}_pred", fontsize=10)
            axes[i].imshow(pred, cmap='gray', vmin=0, vmax=255)
        except Exception as e:
            print(f"{name} failed: {str(e)}")
            axes[i].imshow(np.zeros_like(image))
            axes[i].set_title(f"{name} failed", fontsize=10)
    
    # Hide unused axes
    for j in range(i+1, nrow*ncol):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# Example usage
model_factory = {
    "MobileNetV3": model_mbv3,
    "FastSCNN": model_fast_scnn,
    "BiSeNetV2": model_bisenetv2,
    "U2Net_full": model_u2net_full,
    "U2Net_lite": model_u2net_lite,
}

if __name__ == "__main__":
    test_dir = "/home/william/extdisk/data/motorEV/FC_20250409/Infrared_L_0_calib/"
    comp_dir = "/home/william/extdisk/data/motorEV/FC_20250409/cmp_v3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(comp_dir, exist_ok=True)
    image_paths = [
        f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))
    ]
    for p in tqdm(image_paths):
        image_path = os.path.join(test_dir, p)
        output_path = os.path.join(comp_dir, p.replace(".jpg", ".png"))
        vis_compare(model_factory, image_path, output_path, fixed_size=(480, 640), device=device, transpose=True)
    print("done!")
