import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# datasets for ACE20k sky segmentation
class ACE20kSkyDataset(Dataset):
    def __init__(self, root_dir, split="train", img_size=(480, 640), sky_label=3):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.sky_label = sky_label

        split_dict = {"train": "training", "val": "validation"}
        self.image_dir = os.path.join(root_dir, "images", split_dict[split])
        self.mask_dir = os.path.join(root_dir, "annotations", split_dict[split])

        self.image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        )

        # we will adjust the transfrom from 3-channles grayscale to 1 channel grayscale
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.Grayscale(num_output_channels=1),  # True grayscale
                transforms.ToTensor(),
                # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # [3,H,W]
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=Image.Resampling.NEAREST),
                transforms.PILToTensor(),
            ]
        )

        if self.split == "train":
            self.augment = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(90),
                ]
            )
        else:
            self.augment = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        mask_name = image_name.replace(".jpg", ".png")

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.augment:
            # Set random seed for reproducibility
            seed = torch.randint(0, 2**32, (1,)).item()

            torch.manual_seed(seed)
            image = self.augment(image)

            torch.manual_seed(seed)
            mask = self.augment(mask)
        # Transform image to 3-channel grayscale
        image = self.image_transform(image)

        # Transform and binarize mask
        mask = self.mask_transform(mask)
        binary_mask = (mask.squeeze(0) == self.sky_label).long()  # [H,W]
        return image, binary_mask
    
    def verify_sample(self, idx=0):
        """Debug tool: Checks shapes and consistency"""
        image, mask = self[idx]
        
        print(f"Image shape: {image.shape} (min={image.min():.2f}, max={image.max():.2f})")
        print(f"Mask shape: {mask.shape} (unique values: {torch.unique(mask)})")
        
        # Verify all 3 channels are identical
        assert torch.allclose(image[0], image[1]), "Channels not identical!"
        assert torch.allclose(image[1], image[2]), "Channels not identical!"
        
        return image, mask


def get_dataloader(root_dir, batch_size=8, num_workers=8, pin_memory=True):
    train_dataset = ACE20kSkyDataset(root_dir, split="train")
    val_dataset = ACE20kSkyDataset(root_dir, split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def get_ddp_dataloader(
    root_dir, world_size, rank, batch_size=8, num_workers=8, pin_memory=True
):
    train_dataset = ACE20kSkyDataset(root_dir, split="train")
    val_dataset = ACE20kSkyDataset(root_dir, split="val")

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    dataset = ACE20kSkyDataset("/home/william/extdisk/data/ACE20k/ACE20k_sky")
    image, mask = dataset.verify_sample(idx=0)
    print(f"Sample image shape: {image.shape}")
    print(f"Sample mask shape: {mask.shape}")
    print(torch.unique(image))
    print(torch.unique(mask))
