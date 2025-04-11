import torch
import os
import numpy as np
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data.distributed import DistributedSampler
from multiprocessing import Pool, cpu_count

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def compute_chunk_stats(chunk_paths):
    """Process a chunk of images and return sum, sum_sq, count."""
    chunk_sum = 0
    chunk_sum_sq = 0
    chunk_count = 0

    for img_path in chunk_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue  # Skip corrupt files
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        chunk_sum += np.sum(img)
        chunk_sum_sq += np.sum(img**2)
        chunk_count += img.size

    return chunk_sum, chunk_sum_sq, chunk_count


def parallel_mean_std(dataset_dir, num_workers=None):
    """Compute mean and std using all CPU cores."""
    # Get all image paths
    img_paths = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Split paths into chunks for each worker
    num_workers = num_workers or cpu_count()
    chunk_size = len(img_paths) // num_workers
    chunks = [
        img_paths[i : i + chunk_size] for i in range(0, len(img_paths), chunk_size)
    ]

    # Process chunks in parallel
    with Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(compute_chunk_stats, chunks),
                total=len(chunks),
                desc="Processing images",
            )
        )

    # Aggregate results
    total_sum = sum(r[0] for r in results)
    total_sum_sq = sum(r[1] for r in results)
    total_count = sum(r[2] for r in results)

    mean = total_sum / total_count
    std = np.sqrt((total_sum_sq / total_count) - (mean**2))

    return mean, std


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
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0), p=0.2),
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

        print(
            f"Image shape: {image.shape} (min={image.min():.2f}, max={image.max():.2f})"
        )
        print(f"Mask shape: {mask.shape} (unique values: {torch.unique(mask)})")

        # Verify all 3 channels are identical
        assert torch.allclose(image[0], image[1]), "Channels not identical!"
        assert torch.allclose(image[1], image[2]), "Channels not identical!"

        return image, mask


class ACE20kSkyDatasetV2(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        img_size=(480, 640),
        sky_label=3,
        mu=0.4817,
        sigma=0.2591,
    ):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.sky_label = sky_label

        split_dict = {"train": "training", "val": "validation"}
        self.image_dir = os.path.join(root_dir, "images", split_dict[split])
        self.mask_dir = os.path.join(root_dir, "annotations", split_dict[split])

        # Compute stats only once (cache if needed)
        if mu is None or sigma is None:
            self.mu, self.sigma = parallel_mean_std(self.image_dir, num_workers=8)
            print(f"Computed mean: {self.mu:.4f}, std: {self.sigma:.4f}")
            # Save stats to a file for future use
            stats_file = os.path.join(self.root_dir, "stats.txt")
            with open(stats_file, "w") as f:
                f.write(f"{self.mu:.4f} {self.sigma:.4f}")
        else:
            self.mu = mu
            self.sigma = sigma

        self.image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        )

        # Base transforms (applied always)
        self.image_transform = A.Compose(
            [
                A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_LINEAR),
                A.ToFloat(),
                ToTensorV2(),
            ]
        )
        self.mask_transform = A.Compose(
            [
                A.Resize(img_size[0], img_size[1], interpolation=cv2.INTER_NEAREST),
                ToTensorV2(),
            ]
        )

        # Train-time augmentations (synchronized)
        if self.split == "train":
            self.joint_augment = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Rotate(limit=90, p=0.5),
                ],
                additional_targets={"mask": "image"},
            )

            self.image_augment = A.Compose(
                [
                    A.Normalize(mean=[self.mu], std=[self.sigma]),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
                    A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), p=0.4),
                    A.GaussNoise(p=0.2),
                    # A.RandomGamma(gamma_limit=(70, 130), eps=1e-7, p=0.3), # TODO: need to figure it out why gamma not working
                    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.5, 1.0), p=0.2),
                    A.ElasticTransform(
                        alpha=1,
                        sigma=25,
                        p=0.3,  # Warping (helps generalize edges)
                    ),
                ]
            )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        mask_name = image_name.replace(".jpg", ".png")

        image = cv2.imread(
            os.path.join(self.image_dir, image_name), cv2.IMREAD_GRAYSCALE
        )
        mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_UNCHANGED)

        # Apply synchronized augmentations (flips/rotations)
        if self.split == "train":
            augmented = self.joint_augment(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
            image = self.image_augment(image=image)["image"]  # Image-only augs

        # Resize + ToTensor
        image = self.image_transform(image=image)["image"]
        mask = self.mask_transform(image=mask)["image"]
        binary_mask = (mask.squeeze(0) == self.sky_label).long()

        return image, binary_mask


def get_dataloader(root_dir, batch_size=8, num_workers=8, pin_memory=True, use_v2=True):
    if use_v2:
        train_dataset = ACE20kSkyDatasetV2(root_dir, split="train")
        val_dataset = ACE20kSkyDatasetV2(root_dir, split="val")
    else:
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
    root_dir,
    world_size,
    rank,
    batch_size=8,
    num_workers=8,
    pin_memory=True,
    use_v2=True,
):
    if use_v2:
        train_dataset = ACE20kSkyDatasetV2(root_dir, split="train")
        val_dataset = ACE20kSkyDatasetV2(root_dir, split="val")
    else:
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
