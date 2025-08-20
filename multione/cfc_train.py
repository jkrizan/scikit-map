#%%
#import ncps
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
    #dataset = ArcoV2Dataset(fn_zarr, np.arange(2000, 2024), sequence_length=12)
    #dataset = ArcoV2Dataset.from_arco(Path('/home/josip/arcov2/sample_v1.arco/'),years=np.arange(2000, 2024), n_threads=8, sequence_length=12, limit=100)
    #print(f'Dataset has {len(dataset)} samples, {dataset.n_features} features, {dataset.n_output_bands} output bands, and {dataset.n_timeless_features} timeless features.')

    #factory = ArcoV2DataLoaderFactory(dataset, validation_size=0.2, random_seed=42)

    #train_loader = factory.get_train_loader()
    #val_loader = factory.get_val_loader()

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

    #trainer = MyTrainer(model, train_loader, val_loader, hparams)
    #trainer.train(epochs=10)

# %%

if __name__=="__main__":
    train_test_v1()