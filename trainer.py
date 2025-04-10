import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

import os
import numpy as np
import time
import datetime

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.mobilenetv3_lraspp import lraspp_mobilenet_v3_large
from models.bisenetv2 import bisenetv2
from models.fast_scnn import fast_scnn
from models.u2net import u2net
from losses import MixSoftmaxCrossEntropyOHEMLoss, MultiScaleCrossEntropyLoss, MixedEdgeAwareCrossEntropyLoss
from dataset import get_dataloader, get_ddp_dataloader
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import (
    init_process_group,
    get_rank,
    get_world_size,
    destroy_process_group,
)

MODEL_STATE_DICT_NAME = "model_state_dict"
OPTIMIZER_STATE_DICT_NAME = "optimizer_state_dict"
SCHEDULER_STATE_DICT_NAME = "scheduler_state_dict"
EPOCH_NAME = "epoch"

@dataclass
class SkysegConfig:
    # model
    model: str = "u2net"
    # device
    device: str = "cuda"
    # ddp
    distributed: bool = True
    backend: str = "nccl"
    # data
    data_dir: str = "/home/william/extdisk/data/ACE20k/ACE20k_sky"
    save_dir: str = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models"
    log_dir: str = "/home/william/extdisk/data/ACE20k/ACE20k_sky/logs"
    batch_size: int = 16
    img_size: tuple = (480, 640)
    num_workers: int = 8
    pin_memory: bool = True
    # model
    num_classes: int = 2
    lr: float = 1e-4
    aux: str = "train"  # must be in ['train', 'eval', 'pred']
    u2net_type: str = "full"  # must be in ['full', 'lite']
    num_epochs: int = 30
    # scheduler
    # cosine annealing warm restarts
    T_0: int = (10,)
    T_mult: int = (2,)
    eta_min: float = (1e-5,)
    # continue training
    continue_training: bool = False
    ckpt_path: str = ""

# skyseg_config = SkysegConfig()


def mIoU(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    ious = []
    for cls in range(num_classes):
        pred_cls = preds == cls
        mask_cls = targets == cls
        intersection = (pred_cls & mask_cls).sum()
        union = (pred_cls | mask_cls).sum()
        iou = (intersection + 1e-6) / (union + 1e-6)  # Avoid division by zero
        ious.append(iou)
    return torch.mean(torch.tensor(ious))  # Mean IoU


def get_model(config: SkysegConfig):
    model_name = config.model

    if model_name == "lraspp_mobilenet_v3_large":
        model = lraspp_mobilenet_v3_large(num_classes=config.num_classes)
    elif model_name == "fast_scnn":
        _train_flag = True if config.aux == "train" else False
        model = fast_scnn(num_classes=config.num_classes, aux=_train_flag)
    elif model_name == "bisenetv2":
        model = bisenetv2(num_classes=config.num_classes, aux_mode=config.aux)
    elif model_name == "u2net":
        model = u2net(num_classes=config.num_classes, model_type=config.u2net_type)
    else:
        raise ValueError(f"unsupported model backend type! : {model_name}")

    return model


def get_criterion(config: SkysegConfig):
    model_name = config.model

    if model_name == "lraspp_mobilenet_v3_large":
        # criterion = nn.CrossEntropyLoss()
        criterion = MixedEdgeAwareCrossEntropyLoss()
    elif model_name == "fast_scnn" or model_name == "bisenetv2":
        _train_flag = True if config.aux == "train" else False
        criterion = MixSoftmaxCrossEntropyOHEMLoss(aux=_train_flag, aux_weight=0.4)
    elif model_name == "u2net":
        criterion = MultiScaleCrossEntropyLoss()
    else:
        raise ValueError(f"unsupported model backend type! : {model_name}")

    return criterion


def continue_training(
    model: nn.Module,
    ckpt_path: str,
    device="cuda",
    optimizer: nn.Module = None,
    scheduler: nn.Module = None,
):
    checkpoint = torch.load(ckpt_path, map_location=device)
    # check if checkpoint contains names
    if checkpoint.get(MODEL_STATE_DICT_NAME) is None:
        model.load_state_dict(checkpoint, strict=True)
    else:
        model.load_state_dict(checkpoint[MODEL_STATE_DICT_NAME], strict=True)

    if (
        checkpoint.get(OPTIMIZER_STATE_DICT_NAME) is not None
        and optimizer is not None
    ):
        optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE_DICT_NAME])

    if (
        checkpoint.get(SCHEDULER_STATE_DICT_NAME) is not None
        and scheduler is not None
    ):
        scheduler.load_state_dict(checkpoint[SCHEDULER_STATE_DICT_NAME])

    epoch = checkpoint.get(EPOCH_NAME, 0) + 1 if checkpoint.get(EPOCH_NAME) is not None else 0

    return model, optimizer, scheduler, epoch


class Trainer:
    def __init__(self, config: SkysegConfig):
        self.config = config
        # self.model = lraspp_mobilenet_v3_large(num_classes=config.num_classes)
        self.model_name = config.model
        self.model = get_model(config)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr)
        self.criterion = get_criterion(config)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.T_0[0],
            T_mult=config.T_mult[0],
            eta_min=config.eta_min[0],
        )
        self.best_val_loss = float("inf")
        self.best_val_iou = 0.0
        self.distributed = config.distributed
        self.continue_training = config.continue_training
        if self.distributed:
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(self.device)
            dist.init_process_group(
                backend=config.backend
            )  # Use NCCL for GPU communication
            print("RANK:", os.environ.get("RANK"))  # Global process ID
            print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"))  # GPU index per machine
            print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"))  # Total processes
        else:
            self.device = config.device

        if self.distributed:
            self.train_loader, self.val_loader = get_ddp_dataloader(
                config.data_dir,
                self.world_size,
                self.rank,
                config.batch_size,
                config.num_workers,
                config.pin_memory,
            )
        else:
            self.train_loader, self.val_loader = get_dataloader(
                config.data_dir,
                config.batch_size,
                config.num_workers,
                config.pin_memory,
            )

        if self.distributed:
            self.model = self.model.to(self.local_rank)
            self.criterion = self.criterion.to(self.local_rank)
            self.model = DDP(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank
            )
        else:
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)

        if self.continue_training:
            ckpt_path = config.ckpt_path
            if not os.path.exists(ckpt_path):
                raise ValueError(f"checkpoint path {ckpt_path} does not exist!")
            self.model, self.optimizer, self.scheduler, self.n_iters = continue_training(
                self.model,
                ckpt_path,
                device=self.device,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
            print(f"continue training from {ckpt_path} at epoch {self.n_iters}")

        # get the time for now
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        u2net_model_name = ""
        if self.model_name == "u2net" and self.config.u2net_type == "full":
            u2net_model_name = "u2net_full"
        elif self.model_name == "u2net" and self.config.u2net_type == "lite":
            u2net_model_name = "u2net_lite"

        self.save_dir = os.path.join(
            self.config.save_dir, self.model_name, u2net_model_name, f"run_{timestamp}"
        )
        self.log_dir = os.path.join(
            self.config.log_dir, self.model_name, u2net_model_name, f"log_{timestamp}"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir) if self.rank == 0 else None

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.config.num_epochs)):
            self.n_iters = epoch if not self.continue_training else self.n_iters + 1
            # recording time
            start_time = time.time()
            if self.distributed:
                self.train_loader.sampler.set_epoch(epoch)

            self._train_epoch(epoch)
            val_iou = self._val_epoch(epoch)

            if self.rank == 0 and val_iou > self.best_val_iou:
                self.save_model()
                self.best_val_iou = val_iou
                print(f"New best model saved with IoU: {val_iou:.4f}")
            print(f"Epoch {epoch + 1} took {time.time() - start_time:.2f} seconds")

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            if self.model_name == "lraspp_mobilenet_v3_large":
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            elif self.model_name == "fast_scnn":
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            elif self.model_name == "bisenetv2":
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
            elif self.model_name == "u2net":
                outputs = self.model(images)
                loss0, loss = self.criterion(outputs, masks)
            else:
                raise ValueError(f"unsupported backend! : {self.model_name}")

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            running_loss += loss.item()

            if self.rank == 0:
                running_loss += loss.item()
                if i % 10 == 9:
                    self.writer.add_scalar(
                        "Train/Loss",
                        running_loss / 10,
                        epoch * len(self.train_loader) + i,
                    )
                    print(f"Epoch {epoch + 1}, Iter {i + 1}, Loss: {running_loss / 10}")
                    running_loss = 0.0
        if self.rank == 0:
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
        self.scheduler.step()

    def _val_epoch(self, epoch):
        self.model.eval()
        if self.distributed:
            total_iou = torch.tensor(0.0).to(self.local_rank)
        else:
            total_iou = 0.0
        total_loss = 0.0

        with torch.no_grad():
            for i, (images, masks) in enumerate(self.val_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                if self.model_name == "lraspp_mobilenet_v3_large":
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    preds = torch.argmax(outputs, dim=1)
                elif self.model_name == "fast_scnn":
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    preds = torch.argmax(outputs[0], dim=1)
                elif self.modmodel_nameel == "bisenetv2":
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    preds = torch.argmax(outputs[0], dim=1)
                elif self.model_name == "u2net":
                    outputs = self.model(images)
                    loss0, loss = self.criterion(outputs, masks)
                    preds = torch.argmax(outputs[0], dim=1)
                else:
                    raise ValueError(f"unsupported backend! : {self.model_name}")

                total_loss += loss.item()
                batch_iou = mIoU(preds, masks, self.config.num_classes)
                total_iou += batch_iou

                if self.rank == 0 and (epoch == 0 or epoch % 5 == 4):  # Every 5 epochs
                    if self.writer is not None:
                        self.writer.add_images("val/input", images[:3], epoch)
                        self.writer.add_images(
                            "val/pred", preds[:3].unsqueeze(1).float(), epoch
                        )  # Add channel dim
                        self.writer.add_images(
                            "val/target", masks[:3].unsqueeze(1).float(), epoch
                        )

        if self.distributed:
            dist.all_reduce(total_iou, op=dist.ReduceOp.SUM)
            total_iou = total_iou / self.world_size

        mean_loss = total_loss / len(self.val_loader)
        mean_iou = total_iou / len(self.val_loader)

        if self.rank == 0:
            self.writer.add_scalar("val/loss", mean_loss, epoch)
            self.writer.add_scalar("val/IoU", mean_iou, epoch)
            print(f"Epoch {epoch + 1}, Val Loss: {mean_loss:.4f}, IoU: {mean_iou:.4f}")
        return mean_iou

    def save_model(self):
        if self.rank == 0:
            save_path = os.path.join(
                self.save_dir,
                f"{self.model_name}_{self.n_iters}_iou_{self.best_val_iou:.4f}.pth",
            )
            checkpoint = {
                MODEL_STATE_DICT_NAME: self.model.state_dict(),
                OPTIMIZER_STATE_DICT_NAME: self.optimizer.state_dict(),
                SCHEDULER_STATE_DICT_NAME: self.scheduler.state_dict(),
                EPOCH_NAME: self.n_iters,
            }
            # torch.save(self.model.state_dict(), save_path) # for old runs
            torch.save(checkpoint, save_path)
            print(f"Model saved to {save_path}")

    def close(self):
        if self.rank == 0 and self.writer is not None:
            self.writer.close()
        if self.distributed:
            dist.destroy_process_group()
