import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

import os
import numpy as np
import wandb
import time
import datetime

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.mobilenetv3_lraspp import lraspp_mobilenet_v3_large
from dataset import get_dataloader, get_ddp_dataloader
from dataclasses import dataclass
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, get_rank, get_world_size, destroy_process_group
@dataclass
class SkysegConfig:
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
    num_epochs: int = 30
    save_dir: str = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models"
    # scheduler
    # cosine annealing warm restarts
    T_0: int = 10,
    T_mult: int = 2,
    eta_min: float = 1e-5,
    # wandb
    project: str = "skyseg"
    entity: str = "weiiew"
    run_name: str = "skyseg_mobilenetv3_lraspp"

# skyseg_config = SkysegConfig()

# wandb.init(project=skyseg_config.project, entity=skyseg_config.entity, name=skyseg_config.run_name)

def mIoU(preds:torch.Tensor, targets:torch.Tensor, num_classes:int):
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        mask_cls = (targets == cls)
        intersection = (pred_cls & mask_cls).sum()
        union = (pred_cls | mask_cls).sum()
        iou = (intersection + 1e-6) / (union + 1e-6)  # Avoid division by zero
        ious.append(iou)
    return torch.mean(torch.tensor(ious))  # Mean IoU
   

class Trainer:
    def __init__(self, config: SkysegConfig):
        self.config = config
        self.model = lraspp_mobilenet_v3_large(num_classes=config.num_classes)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=config.T_0[0], T_mult=config.T_mult[0], eta_min=config.eta_min[0])
        # wandb.watch(self.model, log='all')
        self.best_val_loss = float('inf')
        self.best_val_iou = 0.0
        self.distributed = config.distributed
        if self.distributed:
            self.rank = int(os.environ['RANK'])
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.local_rank}'
            torch.cuda.set_device(self.device)
            dist.init_process_group(backend=config.backend)  # Use NCCL for GPU communication
            print("RANK:", os.environ.get("RANK"))          # Global process ID
            print("LOCAL_RANK:", os.environ.get("LOCAL_RANK"))  # GPU index per machine
            print("WORLD_SIZE:", os.environ.get("WORLD_SIZE"))  # Total processes
        else:
            self.device = config.device

        if self.distributed:
            self.train_loader, self.val_loader = get_ddp_dataloader(config.data_dir, self.world_size, self.rank, config.batch_size, config.num_workers, config.pin_memory)
        else:
            self.train_loader, self.val_loader = get_dataloader(config.data_dir, config.batch_size, config.num_workers, config.pin_memory)

        if self.distributed:
            self.model = self.model.to(self.local_rank)
            self.criterion = self.criterion.to(self.local_rank)
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            self.model = self.model.to(self.device)
            self.criterion = self.criterion.to(self.device)
        
        # get the time for now
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.save_dir = os.path.join(self.config.save_dir, f"run_{timestamp}")
        self.log_dir = os.path.join(self.config.log_dir, f"log_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir) if self.rank == 0 else None

    def train(self):
        self.model.train()
        for epoch in tqdm(range(self.config.num_epochs)):
            self.n_iters = epoch
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
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            running_loss += loss.item()

            if self.rank == 0:
                running_loss += loss.item()
                if i % 10 == 9:
                    # wandb.log({"train_loss": running_loss / 10})
                    self.writer.add_scalar("Train/Loss", running_loss / 10, epoch * len(self.train_loader) + i)
                    print(f"Epoch {epoch + 1}, Iter {i + 1}, Loss: {running_loss / 10}")
                    running_loss = 0.0
        if self.rank == 0:
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]['lr'], epoch)
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
                outputs = self.model(images)
                
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                batch_iou = mIoU(preds, masks, self.config.num_classes)
                total_iou += batch_iou

                if self.rank == 0 and (epoch == 0 or epoch % 5 == 4):  # Every 5 epochs
                    if self.writer is not None:
                        self.writer.add_images("val/input", images[:3], epoch)
                        self.writer.add_images("val/pred", preds[:3].unsqueeze(1).float(), epoch)  # Add channel dim
                        self.writer.add_images("val/target", masks[:3].unsqueeze(1).float(), epoch)

        if self.distributed:
            dist.all_reduce(total_iou, op=dist.ReduceOp.SUM)
            total_iou = total_iou / self.world_size
        
        mean_loss = total_loss / len(self.val_loader)
        mean_iou = total_iou / len(self.val_loader)
        # wandb.log({
        #     "val_loss": mean_loss,
        #     "val_iou": mean_iou,
        #     "lr": self.optimizer.param_groups[0]['lr']
        # })
        if self.rank == 0:
            self.writer.add_scalar("val/loss", mean_loss, epoch)
            self.writer.add_scalar("val/IoU", mean_iou, epoch)
            print(f"Epoch {epoch + 1}, Val Loss: {mean_loss:.4f}, IoU: {mean_iou:.4f}")
        return mean_iou

    def save_model(self):
        if self.rank == 0:
            save_path = os.path.join(self.save_dir, f"skyseg_mobilenetv3_lraspp_{self.n_iters}_iou_{self.best_val_iou:.4f}.pth")
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    def close(self):
        if self.rank == 0 and self.writer is not None:
            self.writer.close()
        if self.distributed:
            dist.destroy_process_group()