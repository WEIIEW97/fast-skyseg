import json
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_masks_from_coco(json_path, output_dir, sky_label, show_example=False):
    """
    Generate combined segmentation masks (all annotations per image merged into one mask)
    
    Args:
        json_path: Path to COCO format JSON file
        output_dir: Directory to save output masks
        show_example: Whether to display an example mask (default: False)
    """
    # Load JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Create mapping from image_id to image info
    id_to_image = {img['id']: img for img in data['images']}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Group annotations by image_id
    from collections import defaultdict
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        if ann.get('segmentation'):
            annotations_by_image[ann['image_id']].append(ann)
    
    # Process each image
    for image_id, annotations in tqdm(annotations_by_image.items(), desc="Generating combined masks"):
        # Get image info
        img_info = id_to_image.get(image_id)
        if not img_info:
            continue
            
        # Create blank mask with proper OpenCV-compatible format
        combined_mask = np.zeros((img_info['height'], img_info['width'], 1), dtype=np.uint8)
        
        # Combine all annotations for this image
        for ann in annotations:
            # Process each polygon in segmentation
            for polygon in ann['segmentation']:
                # Ensure polygon is a list of numbers
                if not isinstance(polygon, list):
                    continue
                    
                # Convert to numpy array with proper shape
                try:
                    pts = np.array(polygon, dtype=np.int32)
                    if len(pts.shape) == 1:
                        pts = pts.reshape((-1, 2))
                    pts = pts.reshape((-1, 1, 2))  # Shape required by fillPoly
                    
                    # Draw polygon on mask
                    cv2.fillPoly(combined_mask, [pts], color=(sky_label,))
                except Exception as e:
                    print(f"Error processing polygon in annotation {ann['id']}: {e}")
                    continue
        
        # Remove singleton dimension if present
        if combined_mask.shape[-1] == 1:
            combined_mask = combined_mask.squeeze(-1)
        
        # Generate output filename
        img_name = os.path.splitext(img_info['file_name'])[0]
        mask_name = f"{img_name}.png"
        output_path = os.path.join(output_dir, mask_name)
        combined_mask = np.rot90(combined_mask, k=1)  
        cv2.imwrite(output_path, combined_mask)
        
        # Display example if requested
        if show_example and image_id == list(annotations_by_image.keys())[0]:
            plt.imshow(combined_mask, cmap='gray')
            plt.title(f"Combined Mask\nImage: {img_info['file_name']}\nAnnotations: {len(annotations)}")
            plt.axis('off')
            plt.show()


def imrotate90(image_dir, out_dir=None):
    """
    Rotate images in a directory by 90 degrees clockwise.
    
    Args:
        image_dir (str): Directory containing images to rotate
        out_dir (str, optional): Directory to save rotated images
    """
    if out_dir is None:
        out_dir = image_dir
    
    os.makedirs(out_dir, exist_ok=True)
    
    for img_name in tqdm([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))], desc="Rotating images"):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(os.path.join(out_dir, img_name), rotated_img)
    print(f"Rotated images saved to {out_dir}")


if __name__ == '__main__':
    ROOT_IMG_DIR = "/home/william/extdisk/data/motorEV/FC_20250415/Infrared_L_0_calib/"
    SKY_LABEL = 3

    coco_json_file = "/home/william/extdisk/data/motorEV/FC_20250415/instances_default.json"
    output_dir = "/home/william/extdisk/data/motorEV/FC_20250415/annotations"
    create_masks_from_coco(coco_json_file, output_dir, sky_label=SKY_LABEL, show_example=False)
    imrotate90(ROOT_IMG_DIR, out_dir="/home/william/extdisk/data/motorEV/FC_20250415/images")
