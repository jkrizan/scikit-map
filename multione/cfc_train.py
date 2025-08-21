#%%
#import ncps
from typing import Any
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import gc

#import utils, processing_utils
import time
import matplotlib.pyplot as plt
from pathlib import Path

import pytorch_lightning as pl
from sklearn.externals.array_api_compat import device
import torchmetrics
import torch.nn as nn
import torch

from cfc_dataset import ArcoV2Dataset, ArcoV2DataLoader, ArcoV2DataLoaderFactory
import cfc

fn_zarr = Path(f"/data/oemc/arcov2/sample_v1.zarr")
fn_zarr = Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.zarr")
years = np.arange(2000, 2024)

#%%
'''
import importlib
import cfc
cfc = importlib.reload(cfc)
CfC = cfc.CfC

import importlib
import cfc_dataset
cfc_dataset = importlib.reload(cfc_dataset)
ArcoV2Dataset = cfc_dataset.ArcoV2Dataset
ArcoV2DataLoader = cfc_dataset.ArcoV2DataLoader
ArcoV2DataLoaderFactory = cfc_dataset.ArcoV2DataLoaderFactory

import importlib
import torch
torch=importlib.reload(torch)
nn = torch.nn

'''
#%%
class ArcoV2Learner(pl.LightningModule):
    # https://lightning.ai/docs/pytorch/LTS/common/lightning_module.html

    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self._hparams = hparams

        #self.train_error = torchmetrics.MeanSquaredError()
        #self.valid_error = torchmetrics.MeanSquaredError()

    def forward(self, x, input_timeless, timespans):
        # Only for inference !!
        return self.model(x, input_timeless=input_timeless, timespans=timespans)

    def training_step(self, batch, batch_idx):
        # batch,_ = dataset[0]; (y, x, x_timeless, timespans) = batch
        # data = batch
        idx, (y, x, x_timeless, timespans) = batch

        y_hat, (hc,cx) = self.model.forward(x, input_timeless=x_timeless, hx=None, timespans=timespans)
        y_hat = y_hat.view_as(y)

        #loss = self.train_error(y_hat, y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_mse", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=y.shape[0]) 
        return loss
    
        #loss = nn.MSELoss()(y_hat, y)

        #self.log("train_loss", loss, prog_bar=True)
        #return {"loss": loss}

    def validation_step(self, batch, batch_idx):      

        idx, (y, x, x_timeless, timespans) = batch
        y_hat, _ = self.model.forward(x, input_timeless=x_timeless, hx=None, timespans=timespans)
        y_hat = y_hat.view_as(y)
        
        #loss = self.valid_error(y_hat, y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("val_mse", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=y.shape[0])

        return loss

    # def test_step(self, batch, batch_idx):
    #     # Here we just reuse the validation_step for testing
    #     return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self._hparams['lr'])
    # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._hparams['lr'])

    def prepare_data(self) -> None:
        # This method is called only once?
        print(f'Hello from "prepare_data", GPU: {torch.cuda.current_device()}')


    def setup(self, stage=None):
        # https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/datamodules.html
        # This method is called on every GPU, so we can set up the dataset here
        # This is all run in same time !!!
        # if stage == 'fit' or stage is None:
        #     self.train_loader = ArcoV2DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        #     self.val_loader = ArcoV2DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        print(f'Hello from "setup", {stage=}, GPU: {torch.cuda.current_device()}')
        dataset = ArcoV2Dataset.from_arco(Path('/home/josip/arcov2/sample_v1.arco/'),years=np.arange(2000, 2024), n_threads=8, sequence_length=12, limit=100, device=torch.cuda.current_device())
        factory = ArcoV2DataLoaderFactory(dataset, validation_size=0.2, random_seed=42)

        self.train_loader = factory.get_train_loader()
        self.val_loader = factory.get_val_loader()

    def train_dataloader(self) -> ArcoV2DataLoader:
        return self.train_loader

    def val_dataloader(self) -> ArcoV2DataLoader:
        return self.val_loader

class MyTrainer():
    def __init__(self, model, train_loader, val_loader, hparams):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.hparams = hparams
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            for batch in self.train_loader:
                # loader = iter(self.train_loader)
                # batch = next(loader)
                self.optimizer.zero_grad()
                idx, (y, x, x_timeless, timespans) = batch
                y_hat, (hc, cx) = self.model.forward(x, input_timeless=x_timeless, hx=None, timespans=timespans)
                if y_hat.isnan().any():
                    print(f"NaN values found in predictions for batch {idx}")
                    #break
                y_hat = y_hat.view_as(y)
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()
                print(f"{idx} - Train Loss: {loss.item()}")

            self.validate()

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                idx, (y, x, x_timeless, timespans) = batch
                y_hat, (hc, cx) = self.model.forward(x, input_timeless=x_timeless, hx=None, timespans=timespans)
                y_hat = y_hat.view_as(y)
                loss = self.criterion(y_hat, y)
                print(f"Val Loss: {loss.item()}")

#%%
def train_test_v1():
    #%%
    input_size = 9 #dataset.n_features
    output_size = 7 #dataset.n_output_bands
    n_timeless_features = 17 #dataset.n_timeless_features

    model = cfc.CfC(input_size,
                num_hidden_units=64,
                output_size=output_size,
                timeless_layers=[n_timeless_features, n_timeless_features//2, n_timeless_features//3],
                backbone_layers=[128, 64, 32],
                backbone_dropout=0.1
                )
                
    hparams = {
            'lr': 0.01
            }
    learner = ArcoV2Learner(model, hparams)

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(max_epochs=100, num_nodes=1) #, devices=[0,1])
    trainer.fit(learner) #, train_loader, val_loader)

# %%
from torch import nn

class SimpleModel_v2(nn.Module):
    def __init__(self, output_size):
        super(SimpleModel_v2, self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5),    # torch.Size([1, 64, 5, 8])
            nn.Dropout2d(0.1),
            nn.LazyBatchNorm2d(),   #torch.Size([1, 64, 5, 8])
            nn.ReLU(),
            #nn.AdaptiveAvgPool2d(output_size=)
            nn.LazyConv2d(32, kernel_size=3),   #torch.Size([1, 32, 3, 6])
            nn.LazyBatchNorm2d(),            
            nn.Flatten(),  # torch.Size([1, 32*3*6])            
        )
        self.time=nn.Sequential(
            nn.LazyLinear(32),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.LazyLinear(32),
            nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.LazyLinear(32),
            nn.Flatten()
        )
        self.merge = nn.Sequential(
            nn.LazyLinear(64),
            #nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.LazyLinear(32),
            #nn.LazyBatchNorm1d(),
            nn.ReLU(),
            nn.LazyLinear(output_size)
        )

    def forward(self, x, timespans):
        x_feat = self.conv(x)
        x_time = self.time(timespans)
        x_merge = torch.cat([x_feat, x_time], dim=1)
        return self.merge(x_merge)
    
#%%
from typing import Optional, Union,Any
class SimpleDataLoader:
    def __init__(self, dataset: ArcoV2Dataset, inds: NDArray, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.inds = inds
        self.shuffle = shuffle
        self.length = len(inds)
        self._current_index = 0

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Any:
        if self.shuffle:
            np.random.shuffle(self.inds)
        self._current_index = 0
        return self

    def __next__(self):
        if self._current_index < self.length:
            idx = self.inds[self._current_index]
            self._current_index += 1
            data, _ = self.dataset[idx]
            y, x, _, timespans = data
            x = x.unsqueeze(1)  # Add channel dimension
            timespans = timespans.unsqueeze(1)  # Add channel dimension
            return self._current_index, (y, x, timespans)
        else:
            raise StopIteration

class SimpleLoaderFactory:
    def __init__(self, dataset: ArcoV2Dataset, validation_size: float, random_seed:int):
        self.dataset = dataset
        self.validation_size = validation_size
        self.rs = np.random.RandomState(random_seed)

        all_inds = self.rs.permutation(np.arange(len(dataset)))

        n_val = int(len(dataset) * validation_size)

        self.train_inds = all_inds[:-n_val]
        self.val_inds = all_inds[-n_val:]

    def get_train_loader(self) -> SimpleDataLoader:
        return SimpleDataLoader(self.dataset, self.train_inds)

    def get_val_loader(self) -> SimpleDataLoader:
        return SimpleDataLoader(self.dataset, self.val_inds)
#%%
class SimpleLearner_v2(pl.LightningModule):
    def __init__(self, input_size:int, sequence_length:int, output_size:int, lr:float):
        super(SimpleLearner_v2, self).__init__()
        self.criterion = nn.MSELoss()
        self.model = SimpleModel_v2(output_size)
        _ = self.model.forward(torch.randn(1, 1, sequence_length, input_size), torch.randn(1, 1, sequence_length)) #init Lazy ones   #batch_size, channel_size, height, width
        self.save_hyperparameters()

    def forward(self, x, timespans):
        return self.model(x, timespans)

    def training_step(self, batch, batch_idx):
        _,(y, x, timespans) = batch
        y_hat = self(x, timespans)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=y.shape[0],  sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _,(y, x, timespans) = batch
        y_hat = self(x, timespans)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=y.shape[0],  sync_dist=True)
        return loss

    def setup(self, stage=None):
        # https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/datamodules.html
        # This method is called on every GPU, so we can set up the dataset here
        # This is all run in same time !!!
        # if stage == 'fit' or stage is None:
        #     self.train_loader = ArcoV2DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        #     self.val_loader = ArcoV2DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        print(f'Hello from "setup", {stage=}, GPU: {torch.cuda.current_device()}')
        dataset = ArcoV2Dataset.from_arco(Path('/home/josip/arcov2/sample_v1.arco/'),years=np.arange(2000, 2024), n_threads=8, sequence_length=12, limit=10, device=torch.cuda.current_device())
        factory = SimpleLoaderFactory(dataset, validation_size=0.2, random_seed=42)

        self.train_loader = factory.get_train_loader()
        self.val_loader = factory.get_val_loader()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])

    def train_dataloader(self) -> ArcoV2DataLoader:
        return self.train_loader

    def val_dataloader(self) -> ArcoV2DataLoader:
        return self.val_loader
#%%
def train_test_v2():
    input_size = 9
    output_size = 7
    sequence_length=12

    learner = SimpleLearner_v2(input_size, sequence_length, output_size, lr=0.01)

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(max_epochs=100, num_nodes=1) #, devices=[0,1])
    trainer.fit(learner) #, train_loader, val_loader)

if __name__=="__main__":
    train_test_v2()
# %%
