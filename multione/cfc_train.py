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
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        self._hparams = hparams

    def training_step(self, batch, batch_idx):
        # batch,_ = dataset[0]; (y, x, x_timeless, timespans) = batch
        # data = batch
        idx, (y, x, x_timeless, timespans) = batch

        y_hat, (hc,cx) = self.model.forward(x, input_timeless=x_timeless, hx=None, timespans=timespans)
        y_hat = y_hat.view_as(y)

        loss = nn.MSELoss()(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):      

        idx, (y, x, x_timeless, timespans) = batch
        y_hat, _ = self.model.forward(x, input_timeless=x_timeless, hx=None, timespans=timespans)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self._hparams['lr'])
    # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._hparams['lr'])
    
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
    dataset = ArcoV2Dataset.from_arco(Path('/home/josip/arcov2/sample_v1.arco/'),years=np.arange(2000, 2024), sequence_length=12)
    '''
    import pickle
    import lzma
    pickle.dump(dataset, lzma.open('/data/oemc/arcov2/dataset100.pickle.lzma', 'wb'))
    '''
    #import lzma, pickle
    #dataset1 = pickle.load(lzma.open('/data/oemc/arcov2/dataset100.pickle.lzma', 'rb'))
    factory = ArcoV2DataLoaderFactory(dataset, validation_size=0.2, random_seed=42)

    train_loader = factory.get_train_loader()
    val_loader = factory.get_val_loader()

    input_size = dataset.n_features
    output_size = dataset.n_output_bands
    n_timeless_features = dataset.n_timeless_features

    model = cfc.CfC(input_size,
                num_hidden_units=64,
                output_size=output_size,
                timeless_layers=[n_timeless_features, n_timeless_features//2, n_timeless_features//3],
                backbone_layers=[128, 64, 32],
                backbone_dropout=0.1
                )
                
    hparams = {
            'lr': 0.001
            }
    learner = ArcoV2Learner(model, hparams)

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(max_epochs=10) #, devices=[0,1])
    trainer.fit(learner, train_loader, val_loader)

    #trainer = MyTrainer(model, train_loader, val_loader, hparams)
    #trainer.train(epochs=10)

# %%

if __name__=="__main__":
    train_test_v1()