import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# datasets for ACE20k sky segmentation
class ACE20kSkyDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=(480, 640), sky_label=3):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.sky_label = sky_label

        split_dict = {"train": "training", "val": "validation"}
        self.image_dir = os.path.join(root_dir, "images", split_dict[split])
        self.mask_dir = os.path.join(root_dir, "annotations", split_dict[split])

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")])

        self.image_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.Grayscale(num_output_channels=3), # Ensures 3-channel grayscale
            transforms.ToTensor(),
        ])

        
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=Image.Resampling.NEAREST),
            transforms.ToTensor(),
        ])

        if self.split == "train":
            self.augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(60),
            ])
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
        
        image = self.image_transform(image)
        if self.augment:
            image = self.augment(image)

        mask = self.mask_transform(mask).squeeze(0).long()
        # convert the mask to a binary mask, set mask==3 to 1 and others to 0
        mask = (mask == self.sky_label).long()

        return image, mask


def get_dataloader(root_dir, batch_size=8, num_workers=8, pin_memory=True):
    train_dataset = ACE20kSkyDataset(root_dir, split='train')
    val_dataset = ACE20kSkyDataset(root_dir, split='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader
