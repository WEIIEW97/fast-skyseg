import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import numpy as np
import wandb
import time

from models.mobilenetv3_lraspp import lraspp_mobilenet_v3_large
from dataset import get_dataloader
from dataclasses import dataclass

@dataclass
class SkysegConfig:
    # data
    data_dir: str = "/home/william/extdisk/data/ACE20k/ACE20k_sky"
    batch_size: int = 8
    img_size: tuple = (480, 640)
    num_workers: int = 8
    pin_memory: bool = True
    # model
    num_classes: int = 2
    lr: float = 1e-4
    num_epochs: int = 100
    save_dir: str = "/home/william/extdisk/data/ACE20k/ACE20k_sky/models"
    # scheduler
    # cosine annealing warm restarts
    T_0: int = 10,
    T_mult: int = 2,
    eta_min: float = 1e-5,
    # wandb
    project: str = "skyseg"
    entity: str = "williamwei"
    run_name: str = "skyseg_mobilenetv3_lraspp"

skyseg_config = SkysegConfig()

wandb.init(project=skyseg_config.project, entity=skyseg_config.entity, name=skyseg_config.run_name)

def mIoU(pred:torch.Tensor, target:torch.Tensor, num_classes:int):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return np.nanmean(ious)
   

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lraspp_mobilenet_v3_large(pretrained=True, num_classes=config.num_classes).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader, self.val_loader = get_dataloader(config.data_dir, config.batch_size, config.num_workers, config.pin_memory)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, config.T_0, T_mult=config.T_mult, eta_min=config.eta_min)
        wandb.watch(self.model, log='all')

    def train(self):
        self.model.train()
        for epoch in range(self.config.num_epochs):
            self._train_epoch(epoch)
            self._val_epoch(epoch)

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        for i, (images, masks) in enumerate(self.train_loader):
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                wandb.log({"train_loss": running_loss / 10})
                print(f"Epoch {epoch + 1}, Iter {i + 1}, Loss: {running_loss / 10}")
                running_loss = 0.0

    def _val_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, (images, masks) in enumerate(self.val_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()
            wandb.log({"val_loss": running_loss / len(self.val_loader)})
            print(f"Epoch {epoch + 1}, Val Loss: {running_loss / len(self.val_loader)}")

    def save_model(self):
        save_path = os.path.join(self.config.save_dir, f"skyseg_mobilenetv3_lraspp_{time.strftime('%Y%m%d%H%M')}.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")