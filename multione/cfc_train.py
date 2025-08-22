#%%
#import ncps
from ast import List
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
import cfc, cfc_cell

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
        self.fc=nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=5),    # torch.Size([1, 64, 5, 8])
            nn.Dropout2d(0.1),
            nn.LazyBatchNorm2d(),   #torch.Size([1, 64, 5, 8])
            nn.ReLU(),
            #nn.AdaptiveAvgPool2d(output_size=)
            nn.LazyConv2d(32, kernel_size=3),   #torch.Size([1, 32, 3, 6])
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.Flatten(),  # torch.Size([1, 32*3*6])            
            nn.LazyLinear(128),  # torch.Size([1, 64])
            nn.ReLU(),
            nn.LazyLinear(64),   # torch.Size([1, 32])
            nn.ReLU(),
            nn.LazyLinear(32),  # torch.Size([1, 16])
            nn.ReLU(),
            nn.LazyLinear(output_size)  # torch.Size([1, 7])
        )
        # self.time=nn.Sequential(
        #     nn.LazyLinear(32),
        #     nn.LazyBatchNorm1d(),
        #     nn.ReLU(),
        #     nn.LazyLinear(32),
        #     nn.LazyBatchNorm1d(),
        #     nn.ReLU(),
        #     nn.LazyLinear(32),
        #     nn.Flatten()
        # )
        # self.merge = nn.Sequential(
        #     nn.LazyLinear(64),
        #     #nn.LazyBatchNorm1d(),
        #     nn.ReLU(),
        #     nn.LazyLinear(32),
        #     #nn.LazyBatchNorm1d(),
        #     nn.ReLU(),
        #     nn.LazyLinear(output_size)
        # )

    def forward(self, x, timespans):
        ts = timespans.unsqueeze(3).expand_as(x)
        x = torch.cat([x, ts], dim=1)
        
        #x_time = self.time(timespans)
        #x_merge = torch.cat([x_feat, x_time], dim=1)
        return self.fc(x) #self.merge(x_merge)

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
            x[:,:,7] = x[:,:,7] / 10000  # Normalize MS data
            x[:,:,8]  = x[:,:,8] / 100  # Clip GTemp data to max 30
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
    def __init__(self, input_size:int, hidden_size:int, sequence_length:int, output_size:int, backbone_layers, limit:int=100, lr:float=0.01, debug=False):
        super(SimpleLearner_v2, self).__init__()
        self.criterion = nn.MSELoss()

        self.model = CfcModel_v3(input_size=input_size, 
                                hidden_size=hidden_size, 
                                sequence_length=sequence_length, 
                                output_size=output_size, 
                                backbone_layers=backbone_layers, 
                                backbone_dropout=0.1,
                                activation='relu')
        
        #self.model = SimpleModel_v2(output_size)
        #_ = self.model.forward(torch.randn(1, 1, sequence_length, input_size), torch.randn(1, 1, sequence_length)) #init Lazy ones   #batch_size, channel_size, height, width
        self.save_hyperparameters(ignore=['model'])

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

    def fake_dataloader(self,n:int):
        device = torch.cuda.current_device()
        t = torch.randn(2, device=device)
        #print(f"Fake dataloader: {t.device}")
        return (
            (i,
             (torch.randn(2, self.hparams['output_size'], device=device),
              torch.randn(2, 1, self.hparams['sequence_length'], self.hparams['input_size'], device=device),
              torch.randn(2, 1, self.hparams['sequence_length'], device=device)))
            for i in range(n)
        )

    def setup(self, stage=None):
        # https://lightning.ai/docs/pytorch/stable/notebooks/lightning_examples/datamodules.html
        # This method is called on every GPU, so we can set up the dataset here
        # This is all run in same time !!!
        # if stage == 'fit' or stage is None:
        #     self.train_loader = ArcoV2DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        #     self.val_loader = ArcoV2DataLoader(self.val_dataset, batch_size=32, shuffle=False)
        # self.model = CfcModel_v3(self.hparams['input_size'], 
        #                         hidden_size=self.hparams['hidden_size'], 
        #                         sequence_length=self.hparams['sequence_length'], 
        #                         output_size=self.hparams['output_size'], 
        #                         backbone_layers=self.hparams['backbone_layers'], 
        #                         backbone_dropout=0.1,
        #                         activation='relu')
        
        # !!!!!!!
        #print(f"Setup, prije: device={self.model.fc.weight.device}, {self.model.rnn_sequence[0].ff1.weight.device}")
        self.model.transfer_to_device(torch.cuda.current_device())
        
        #print(f"Setup, poslije: device={self.model.fc.weight.device}, {self.model.rnn_sequence[0].ff1.weight.device}")

        if self.hparams['debug']:
            #print(f'Hello from "setup", {stage=}, GPU: {torch.cuda.current_device()}')     
            self.train_loader = self.fake_dataloader(10)
            self.val_loader = self.fake_dataloader(5)
        else:
            dataset = ArcoV2Dataset.from_arco(Path('/home/josip/arcov2/sample_v1.arco/'),years=np.arange(2000, 2024),
                                            n_threads=8, sequence_length=12, limit=self.hparams['limit'],
                                            device=torch.cuda.current_device())
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
#%%
class CfcModel_v3(nn.Module):
    def __init__(self, 
                 input_size:int, 
                 hidden_size: int, 
                 activation: str = 'lecun_tanh', #silu, relu, tanh, gelu, lecun_tanh
                 sequence_length:int = 12, 
                 backbone_layers:list[int]=[128, 64, 32],
                 backbone_dropout: float = 0.1, 
                 output_size:int = 7):
        
        super(CfcModel_v3, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.output_size = output_size
        self.activation=activation

        self.mode='default'

        t=torch.randn(1)    
        #print(f"CfcModel_v3 init device: {t.device}")
        
        self.rnn_sequence = [ 
            cfc_cell.CfCCell(
                self.input_size,
                self.hidden_size,
                self.mode,
                self.activation,
                self.backbone_layers,
                self.backbone_dropout,
                )
                for _ in range(self.sequence_length)
            ]

        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)  # Mixed memory
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        print(f"CfcModel_v3: device={self.fc.weight.device}, {self.rnn_sequence[0].ff1.weight.device}")
        self.init_weights()
        
    def transfer_to_device(self, device):
        self.fc = self.fc.to(device)
        self.lstm = self.lstm.to(device)
        self.rnn_sequence = [cell.to(device) for cell in self.rnn_sequence]

    def forward(self, x, timespans, hx=None):
        # x (batch, 1, seq_len, input_size)
        device = x.device
        x = x.squeeze(1)  # (batch, seq_len, input_size)
        timespans = timespans.squeeze(1)  # (batch, seq_len)
        batch_size, seq_len = x.size(0), x.size(1)

        if hx is None:
            h_state = torch.zeros((batch_size, self.hidden_size), device=device)
            c_state = torch.zeros((batch_size, self.hidden_size), device=device)
        else:
            h_state, c_state = hx
        

        for t in range(seq_len):
            inputs = x[:, t]
            
            ts = 1.0 if timespans is None else timespans[:, t].reshape(-1,1) #.squeeze()

            h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out, h_state = self.rnn_sequence[t].forward(inputs, h_state, ts)

        readout = self.fc(h_out) #type: ignore
        #hx = (h_state, c_state) #if self.use_mixed else h_state

        return readout #, hx


    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)
            else:
                torch.nn.init.uniform_(w)

#%%
def train_test_v2():
    input_size = 9
    output_size = 7
    sequence_length=12

    learner = SimpleLearner_v2(input_size, sequence_length, output_size, limit=100, lr=0.1)

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(max_epochs=100, num_nodes=1) #, devices=[0,1])
    trainer.fit(learner) #, train_loader, val_loader)

def train_test_v3():
    input_size = 9
    output_size = 7
    sequence_length=12

    #torch.set_default_device('cuda')

    # model = CfcModel_v3(input_size, hidden_size=128, sequence_length=sequence_length, output_size=output_size, 
    #                     backbone_layers=[32], backbone_dropout=0.1,
    #                     activation='relu')

    learner = SimpleLearner_v2(input_size, 
                               hidden_size=128, 
                               sequence_length=sequence_length, 
                               output_size=output_size, 
                               backbone_layers=[32],
                               limit=100, 
                               lr=0.1,
                               debug=False)

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(max_epochs=100, num_nodes=1) #, devices=[0,1])
    trainer.fit(learner) #, train_loader, val_loader)



    

if __name__=="__main__":
    #train_test_v2()
    train_test_v3()
# %%
