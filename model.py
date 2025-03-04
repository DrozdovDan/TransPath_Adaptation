import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
from einops import rearrange
import lightning as L
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Any
import wandb
import multiprocessing
import matplotlib.pyplot as plt

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.dropout = nn.Dropout(dropout)
        self.silu = nn.SiLU()
        self.idConv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        h = x

        h = self.norm1(h)
        h = self.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x = self.idConv(x)

        return x + h
    
class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode="nearest")
        x = self.conv(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, downsample_steps, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(
                in_channels, 
                hidden_channels, 
                kernel_size=5, 
                stride=1, 
                padding=2
            )
        ])
        for _ in range(downsample_steps):
            self.layers.append(
                nn.Sequential(
                    ResNetBlock(hidden_channels, hidden_channels, dropout),
                    Downsample(hidden_channels)
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, upsample_steps, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(upsample_steps):
            self.layers.append(
                nn.Sequential(
                    ResNetBlock(hidden_channels, hidden_channels, dropout),
                    Upsample(hidden_channels)
                )
            )
        self.norm = nn.GroupNorm(num_groups=32, num_channels=hidden_channels, eps=1e-6, affine=True)
        self.silu = nn.SiLU()
        self.conv_out = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.silu(x)
        x = self.conv_out(x)
        return torch.tanh(x)
    
def build_grid(resolution, max_v=1.):
    """
    :param resolution: tuple of 2 numbers
    :return: grid for positional embeddings built on input resolution
    """
    ranges = [np.linspace(0., max_v, num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return np.concatenate([grid, max_v - grid], axis=-1)

class PosEmbeds(nn.Module):
    def __init__(self, hidden_size, resolution):
        super().__init__()
        self.linear = nn.Linear(4, hidden_size)
        self.grid = nn.Parameter(Tensor(build_grid(resolution)), requires_grad=False)
        
    def forward(self, inputs):
        pos_emb = self.linear(self.grid).moveaxis(3, 1)
        return inputs + pos_emb
    
    def change_resolution(self, resolution, max_v):
        self.grid = nn.Parameter(Tensor(build_grid(resolution, max_v)), requires_grad=False)

class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=None, dropout=0.2):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class BasicTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.3, context_dim=None):
        super().__init__()
        
        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, kdim=context_dim, vdim=context_dim, batch_first=True)
        self.ff = FeedForward(embed_dim, embed_dim * 4, dropout=dropout)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, context=None):
        h1 = self.norm1(x)
        h1, _ = self.attn1(h1, h1, h1, need_weights=False)
        x = h1 + x
        
        h2 = self.norm2(x)
        context = h2 if context is None else context
        h2, _ = self.attn2(h2, context, context, need_weights=False)
        x = h2 + x

        h3 = self.norm3(x)
        x = self.ff(h3) + x

        return x
    
class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, num_heads,
                 depth=4, dropout=0.3, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(in_channels, num_heads, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        f = x
        f = self.norm(f)
        f = rearrange(f, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            f = block(f, context=context)
        f = rearrange(f, 'b (h w) c -> b c h w', h=h, w=w)
        return f + x
    
class TransPathModel(nn.Module):
    def __init__(self, 
                in_channels=2, 
                out_channels=1, 
                hidden_channels=64,
                attn_blocks=4,
                attn_heads=4,
                cnn_dropout=0.15,
                attn_dropout=0.15,
                downsample_steps=3, 
                resolution=(64, 64)):
        super().__init__()
        
        self.encoder = Encoder(in_channels, hidden_channels, downsample_steps, cnn_dropout)
        self.decoder = Decoder(hidden_channels, out_channels, downsample_steps, cnn_dropout)
        
        self.encoder_pos = PosEmbeds(
            hidden_channels, 
            (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps)
        )
        self.decoder_pos = PosEmbeds(
            hidden_channels, 
            (resolution[0] // 2**downsample_steps, resolution[1] // 2**downsample_steps)
        )

        self.transformer = SpatialTransformer(
            hidden_channels, 
            attn_heads,
            attn_blocks, 
            attn_dropout
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.encoder_pos(x)
        x = self.transformer(x)
        x = self.decoder_pos(x)
        x = self.decoder(x)
        return x
    
class TransPathLit(L.LightningModule):
    def __init__(self, model: nn.Module, mode: str='f', learning_rate: float=1e-4, weight_decay: float=0.0) -> None:
        super().__init__()
        self.save_hyperparameters()
        
        self.model = model
        self.mode = mode
        self.loss = nn.L1Loss() if mode == 'h' else nn.MSELoss()
        self.k = 64*64 if mode == 'h' else 1
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT:
        map_design, start, goal, gt_hmap = batch
        inputs = torch.cat([map_design, start + goal], dim=1) if self.mode in ('f', 'nastar') else torch.cat([map_design, goal], dim=1)
        predictions = self.model(inputs)
        loss = self.loss((predictions + 1) / 2 * self.k, gt_hmap)
        self.log(f'train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ) -> STEP_OUTPUT | None:
        map_design, start, goal, gt_hmap = batch
        inputs = torch.cat([map_design, start + goal], dim=1) if self.mode in ('f', 'nastar') else torch.cat([map_design, goal], dim=1)
        predictions = self.model(inputs)
        loss = self.loss((predictions + 1) / 2 * self.k, gt_hmap)
        self.log(f'val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
    
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