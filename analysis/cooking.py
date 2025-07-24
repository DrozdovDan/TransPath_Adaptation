import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
import torch
import yaml
import sys
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from einops import rearrange
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger, CometLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Any
import wandb
import multiprocessing
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import LearningRateMonitor
from model import TransPathModel, MambaPathModel, MambaPathModel2

def maskedMSELoss(prediction, target, mask):
    N = torch.sum(mask)
    if N == 0:
        return torch.sum((prediction - target) ** 2) * 0.0
    loss = torch.sum((prediction - target) ** 2) / N
    return loss

# Training Module
class TransPathLit(L.LightningModule):
    def __init__(self, model: nn.Module, mode: str='f', learning_rate: float=1e-4, weight_decay: float=0.0, flag_OneCycle = True) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.model = model
        self.mode = mode
        self.loss = nn.L1Loss() if mode == 'h' else maskedMSELoss
        self.k = 64*64 if mode == 'h' else 1
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.flag_OneCycle = flag_OneCycle

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        map_design, start, goal, gt_hmap = batch
        inputs = torch.cat([map_design, start + goal], dim=1) if self.mode in ('f', 'nastar') else torch.cat([map_design, goal], dim=1)
        predictions = self.model(inputs)
        loss = self.loss((predictions + 1) / 2 * self.k * (1 - map_design - goal) + goal, gt_hmap, 1 - map_design - goal)
        self.log(f'train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT | None:
        map_design, start, goal, gt_hmap = batch
        inputs = torch.cat([map_design, start + goal], dim=1) if self.mode in ('f', 'nastar') else torch.cat([map_design, goal], dim=1)
        predictions = self.model(inputs)
        loss = self.loss((predictions + 1) / 2 * self.k * (1 - map_design - goal) + goal, gt_hmap, 1 - map_design - goal)
        self.log(f'val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.flag_OneCycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches
            )
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',           # 'min' для loss, 'max' для accuracy
                factor=0.5,           # во сколько раз уменьшать LR
                patience=2,          # сколько эпох ждать без улучшений
                verbose=True,         # выводить сообщения об изменении LR
                min_lr=1e-10          # минимальный LR
                )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"  # обязательно нужно указать метрику
            }
        }


class PathLogger(L.Callback):
    def __init__(self, val_batch, num_samples=10, mode='f'):
        super().__init__()
        map_design, start, goal, gt_hmap = val_batch[:num_samples]
        inputs = torch.cat([map_design, start + goal], dim=1) if mode == 'f' else torch.cat([map_design, goal], dim=1)
        self.val_samples = inputs[:num_samples]
        if mode == 'f':
            self.hm = gt_hmap[:num_samples]
        elif mode == 'h':
            self.hm =  (gt_hmap / gt_hmap.amax(dim=(2, 3), keepdim=True))[:num_samples]
        else:
            self.hm = gt_hmap[:num_samples]

    def on_validation_epoch_end(self, trainer, lightning_module):
        val_samples = self.val_samples.to(device=lightning_module.device)
        prediction = (lightning_module.model(val_samples) + 1) / 2
        if lightning_module.mode == 'h':
            prediction = prediction * 64 * 64

        trainer.logger.experiment.log({
            'data': [wandb.Image(x) for x in torch.cat([self.val_samples, self.hm], dim=1)],
            'predictions': [wandb.Image(x) for x in torch.cat([val_samples, prediction], dim=1)]
        })


# Dataset
class GridData(Dataset):
    """
    'mode' argument defines type of ground truth values:
        f - focal values
        h - absolute ideal heuristic values
        cf - correction factor values
    """
    def __init__(self, path, mode='f', clip_value=0.95, img_size=64):
        self.img_size = img_size
        self.clip_v = clip_value
        self.mode = mode

        self.maps   = np.load(os.path.join(path,    'maps.npy'),    mmap_mode='c')
        self.goals  = np.load(os.path.join(path,    'goals.npy'),   mmap_mode='c')
        self.starts = np.load(os.path.join(path,    'starts.npy'),  mmap_mode='c')
        
        file_gt = {'f' : 'focal.npy', 'h':'abs.npy', 'cf': 'cf.npy'}[mode]
        self.gt_values = np.load(os.path.join(path, file_gt), mmap_mode='c')

    def __len__(self):
        return len(self.gt_values)
    
    def __getitem__(self, idx):
        gt_ = torch.from_numpy(self.gt_values[idx].astype('float32'))
        if self.mode == 'f':
            gt_=  torch.where( gt_ >= self.clip_v, gt_ , torch.zeros_like( torch.from_numpy(self.gt_values[idx])))
        return (torch.from_numpy(self.maps[idx].astype('float32')), 
                torch.from_numpy(self.starts[idx].astype('float32')), 
                torch.from_numpy(self.goals[idx].astype('float32')), 
                gt_ )


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Configuration
if __name__ == "__main__":
    config = load_config(sys.argv[1])  # или sys.argv[1] для указания при запуске
    model_name          = config["model_name"]
    mode                = config["mode"]
    dataset             = config["dataset"]
    batch_size          = config["batch_size"]
    max_epochs          = config["max_epochs"]
    learning_rate       = config["learning_rate"]
    weight_decay        = config["weight_decay"]
    limit_train_batches = config["limit_train_batches"]
    limit_val_batches   = config["limit_val_batches"]
    proj_name           = config["proj_name"]
    flag                = config["flag"]
    accelerator         = config["accelerator"]
    devices             = config["devices"]
    checkpoints_dir     = config.get("checkpoints_dir")
    weights_dir         = config["weights_dir"]
    checkpoint          = config["checkpoint"]
    continue_learning: bool   = config["continue_learning"]
    img_size: int         = config["img_size"]
    skip: bool              = config["skip"]
    downsample_steps: bool  = config["downsample_steps"]
    embeddings: bool        = config["embeddings"]
    report_to               = config["report_to"]
    run_name            = f"model_name={model_name}, ds={dataset}, bs={batch_size}, ep={max_epochs}, lr={learning_rate}, OneCycle={flag}, skip={skip}, downsample_steps={downsample_steps}, embeddings={embeddings}"

    torch.set_default_device(torch.device(f"cuda:{devices[-1]}"))

    # Load datasets
    dataset_dir = f'{dataset}'
    train_data = GridData(path=f"{dataset_dir}/train", mode=mode, img_size=img_size)
    val_data = GridData(path=f"{dataset_dir}/val", mode=mode, img_size=img_size)
    resolution = (train_data.img_size, train_data.img_size)

    # Create dataloaders
    torch.manual_seed(42)
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=5,  # Uncomment if not in jupyter
        pin_memory=True,
        generator=torch.Generator(device=f'cuda:{devices[-1]}'),
    )
    val_dataloader = DataLoader(
        val_data, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=5,  # Uncomment if not in jupyter
        pin_memory=True
    )
    samples = next(iter(val_dataloader))

    # Initialize model and trainer
    date = time.strftime("%d%m%Y_%H%M%S")
    callback = PathLogger(samples, mode=mode, num_samples=10)
    checkpoints = ModelCheckpoint(dirpath=checkpoints_dir, 
                                  filename=f'{run_name}-ep{{epoch}}-{{val_loss:.5f}}-date:{date}', 
                                  every_n_epochs=1, 
                                  auto_insert_metric_name=False,
                                  monitor="val_loss",
                                  mode="min",
                                  save_top_k=5, 
                                  save_weights_only=False, 
                                  verbose=True,
                                  )
    if report_to == 'wandb':
        wandb_logger = WandbLogger(project=proj_name, name=f'{run_name}_{mode}', log_model='all')
    elif report_to == 'tensorboard':
        tb_logger = TensorBoardLogger("tb_logs", name=f'{run_name}_{mode}')
    elif report_to == "comet_ml":
        comet_logger = CometLogger(api_key=os.environ.get("COMET_API_KEY"), workspace=os.environ.get("COMET_WORKSPACE"), project_name=proj_name, experiment_name=f'{run_name}_{mode}')

    # Initialize model
    if model_name == 'MambaPathModel':
        model = MambaPathModel(resolution=resolution, skip=skip, downsample_steps=downsample_steps, embeddings=embeddings)
    elif model_name == 'MambaPathModel2':
        model = MambaPathModel2(resolution=resolution)
    else:
        model = TransPathModel(resolution=resolution, skip=skip, downsample_steps=downsample_steps, embeddings=embeddings)

    lit_module = TransPathLit(
        model=model,
        mode=mode,
        learning_rate=learning_rate,
        weight_decay=weight_decay, 
        flag_OneCycle=flag
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")   # или "step"

    if report_to == 'wandb':
        trainer = L.Trainer(
            logger=wandb_logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            deterministic=False,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            callbacks=[checkpoints, callback, lr_monitor],
        )
    elif report_to == 'tensorboard':
        trainer = L.Trainer(
            logger=tb_logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            deterministic=False,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            callbacks=[checkpoints, lr_monitor],
        )
    elif report_to == "comet_ml":
        trainer = L.Trainer(
            logger=comet_logger,
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            deterministic=False,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            callbacks=[checkpoints, lr_monitor],
        )
    else:
        trainer = L.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            deterministic=False,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            callbacks=[checkpoints, lr_monitor],
        )


    if continue_learning:
        trainer.fit(lit_module, train_dataloader, val_dataloader, ckpt_path=checkpoint)
    else:
        trainer.fit(lit_module, train_dataloader, val_dataloader)
    
    # Save model weights
    model_name = f'{proj_name}_{run_name}'
    weights_path = os.path.join(weights_dir, model_name + ".ckpt")
    trainer.save_checkpoint(weights_path)
    print(f"Model saved as {os.path.join(weights_dir, model_name)}")