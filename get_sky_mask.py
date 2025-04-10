import glob
import os
import pathlib
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image as pil
import torch
from tqdm import tqdm


def get_images_with_sky_class(
    image_root_path: pathlib.Path, sky_class: int
) -> List[pathlib.Path]:
    """Return a list of all images with sky class `sky_class`."""
    images = []
    for image_path in tqdm(glob.glob(os.path.join(image_root_path, "*.png"))):
        segmentation_mask = pil.open(image_path)
        segmentation_mask = np.array(segmentation_mask)
        if np.any(segmentation_mask == sky_class):
            # get basename of image path without extension
            image_path = os.path.splitext(os.path.basename(image_path))[0]
            images.append(image_path)
    return images


def get_data_splits(
    train_image_root_path: pathlib.Path,
    test_image_root_path: pathlib.Path,
    sky_class: int,
    train_split: float = 0.8,
    seed: int = 42,
) -> Dict[str, List[pathlib.Path]]:
    """Return a tuple of lists of images for train and test split."""
    images = get_images_with_sky_class(train_image_root_path, sky_class)
    # fix random seed for reproducibility
    np.random.seed(seed)
    np.random.shuffle(images)
    train_size = int(len(images) * train_split)
    train_images = images[:train_size]
    validation_images = images[train_size:]
    test_images = get_images_with_sky_class(test_image_root_path, sky_class)

    splits = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images,
    }
    return splits


class SkySegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        image_root_path: pathlib.Path,
        segmentation_root_path: pathlib.Path,
        images: List[pathlib.Path],
        sky_class: int,
        stage: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.image_root_path = image_root_path
        self.segmentation_root_path = segmentation_root_path
        self.images = images
        self.transform = transform
        self.sky_class = sky_class
        self.stage = stage

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, Tuple[int, int]]]:
        image_path = os.path.join(self.image_root_path, self.images[idx] + ".jpg")
        image = pil.open(image_path).convert("RGB")
        image_size = image.size
        image = np.array(image)

        segmentation_path = os.path.join(
            self.segmentation_root_path, self.images[idx] + ".png"
        )
        segmentation_mask = pil.open(segmentation_path)
        # convert the segmentation mask to a binary mask with 1 for sky and 0 for
        # everything else
        segmentation_mask = np.array(segmentation_mask)
        segmentation_mask = np.where(
            segmentation_mask == self.sky_class,
            np.ones_like(segmentation_mask),
            np.zeros_like(segmentation_mask),
        )

        if self.transform and self.stage != "test":
            transformed = self.transform(image=image, mask=segmentation_mask)
            image = transformed["image"]
            segmentation_mask = transformed["mask"].unsqueeze(0)
            return {
                "image": image,
                "mask": segmentation_mask,
            }
        else:
            image = self.transform(image=image)["image"]
            segmentation_mask = torch.from_numpy(segmentation_mask).unsqueeze(0)
            return {
                "image": image,
                "mask": segmentation_mask,
                "image_size": image_size,
            }


if __name__ == "__main__":
    train_root_dir = "/home/william/extdisk/data/ACE20k/ADEChallengeData2016/images/training"
    val_root_dir = "/home/william/extdisk/data/ACE20k/ADEChallengeData2016/images/validation"
    
    train_anno_dir = "/home/william/extdisk/data/ACE20k/ADEChallengeData2016/annotations/training"
    val_anno_dir = "/home/william/extdisk/data/ACE20k/ADEChallengeData2016/annotations/validation"

    sky_class = 3
    sky_export_dir = "/home/william/extdisk/data/ACE20k/ACE20k_sky"
    os.makedirs(sky_export_dir, exist_ok=True)
    sky_anno_export_dir = "/home/william/extdisk/data/ACE20k/ACE20k_sky/annotations"
    os.makedirs(sky_anno_export_dir, exist_ok=True)
    os.makedirs(sky_anno_export_dir + "/training", exist_ok=True)
    os.makedirs(sky_anno_export_dir + "/validation", exist_ok=True)
    sky_image_export_dir = "/home/william/extdisk/data/ACE20k/ACE20k_sky/images"
    os.makedirs(sky_image_export_dir, exist_ok=True)
    os.makedirs(sky_image_export_dir + "/training", exist_ok=True)
    os.makedirs(sky_image_export_dir + "/validation", exist_ok=True)

    sky_train_names = get_images_with_sky_class(train_anno_dir, sky_class)
    sky_val_names = get_images_with_sky_class(val_anno_dir, sky_class)

    # copy sky_train_names, sky_val_names to sky_export_dir
    for name in tqdm(sky_train_names):
        os.system(f"cp {train_root_dir}/{name}.jpg {sky_image_export_dir}/training")
        os.system(f"cp {train_anno_dir}/{name}.png {sky_anno_export_dir}/training")
        
    for name in tqdm(sky_val_names):
        os.system(f"cp {val_root_dir}/{name}.jpg {sky_image_export_dir}/validation")
        os.system(f"cp {val_anno_dir}/{name}.png {sky_anno_export_dir}/validation")
    print("Done")

    import gc
    gc.collect()