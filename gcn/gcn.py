from pathlib import Path
from typing import List

import numpy as np
import torch
import pytorch_lightning as pl
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint



# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────

class GridDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = ".", batch_size: int = 128):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.train_ds: List[Data] | None = None
        self.val_ds: List[Data] | None = None

    def prepare_data(self):
        # nothing to download, but Trigger for DDP
        pass

    def setup(self, stage: str | None = None):
        train_path = "/home/silvarum/TransPath_Adaptation/gcn/datasets/pfu_small/train2.pt"
        val_path   = "/home/silvarum/TransPath_Adaptation/gcn/datasets/pfu_tiny/val.pt"
        self.train_ds, self.val_ds = torch.load(train_path), torch.load(val_path)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)



import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class GCNModule(pl.LightningModule):
    def __init__(self, in_feats: int = 5, hidden: int = 64, 
                 out_feats: int = 1, num_layers: int = 16, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Собираем слой за слоем
        layers = []
        # Первый слой: in_feats -> hidden
        layers.append(GCNConv(in_feats, hidden))
        # Промежуточные: hidden -> hidden
        for _ in range(num_layers - 2):
            layers.append(GCNConv(hidden, hidden))
        # Последний слой: hidden -> out_feats
        layers.append(GCNConv(hidden, out_feats))
        self.convs = nn.ModuleList(layers)

        self.loss_fn = nn.MSELoss()

    def forward(self, data: Data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # Прогоняем через все conv-слои
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight)
            x = F.relu(x)
        # Последний слой без активации (и убираем размерность)
        x = self.convs[-1](x, edge_index, edge_weight).squeeze(-1)
        return x

    def _step(self, batch: Data, stage: str):
        preds = self(batch)
        loss = self.loss_fn(preds, batch.y)
        self.log(f"{stage}_mse", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: Data, batch_idx: int):
        return self._step(batch, "train")

    def validation_step(self, batch: Data, batch_idx: int):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

dm = GridDataModule(data_dir="/home/silvarum/TransPath_Adaptation/gcn_dataset", batch_size=128)

model = GCNModule(hidden=64, lr=1e-3)

logger = WandbLogger(project="gcn-cf", name="16_layers_small", resume="never", reinit=True)

ckpt_cb = ModelCheckpoint(
    monitor="val_mse",          # метрика, за которой следим
    mode="min",                 # «меньше — лучше»
    dirpath="checkpoints/",     # куда класть файлы
    filename="best-{epoch}-{val_mse:.4f}",
    save_top_k=3,               # хранить только лучший
)
trainer = pl.Trainer(
    max_epochs=100,
    logger=logger,
    accelerator="cuda",
    devices=[6],
    callbacks=[ckpt_cb],        # ← добавили
)

trainer.fit(model, datamodule=dm)