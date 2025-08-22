# Copyright 2022 Mathias Lechner and Ramin Hasani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch import nn
from typing import List, Optional, Union
import ncps
from cfc_cell import CfCCell #, WiredCfCCell
import pytorch_lightning as pl
from cfc_dataset import ArcoV2DatasetV2
from torch.utils.data import DataLoader

class CfcModel_v4(nn.Module):
    def __init__(self, 
                 input_size:int, 
                 hidden_size: int, 
                 activation: str = 'lecun_tanh', #silu, relu, tanh, gelu, lecun_tanh
                 sequence_length:int = 12, 
                 backbone_layers:list[int]=[128, 64, 32],
                 backbone_dropout: float = 0.1, 
                 output_size:int = 7):

        super(CfcModel_v4, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.output_size = output_size
        self.activation=activation

        self.mode='default'

        #t=torch.randn(1)    
        #print(f"CfcModel_v3 init device: {t.device}")
        
        self.rnn_sequence = nn.ModuleList( [ 
            CfCCell(
                self.input_size,
                self.hidden_size,
                self.mode,
                self.activation,
                self.backbone_layers,
                self.backbone_dropout,
                )
                for _ in range(self.sequence_length)
            ])

        self.lstm = nn.LSTMCell(self.input_size, self.hidden_size)  # Mixed memory
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        #print(f"CfcModel_v3: device={self.fc.weight.device}, {self.rnn_sequence[0].ff1.weight.device}")
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



        

class CfcLearner_v4(pl.LightningModule):
    def __init__(self, fn_zarr, years, input_size:int, hidden_size:int, sequence_length:int, output_size:int, backbone_layers, limit:int=100, lr:float=0.01, debug=False):
        super(CfcLearner_v4, self).__init__()
        self.criterion = nn.MSELoss()

        self.model = CfcModel_v4(input_size=input_size, 
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
        device = f'cuda: {torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        self.model.transfer_to_device(device)

        #print(f"Setup, poslije: device={self.model.fc.weight.device}, {self.model.rnn_sequence[0].ff1.weight.device}")

        if self.hparams['debug']:
            #print(f'Hello from "setup", {stage=}, GPU: {torch.cuda.current_device()}')     
            self.train_loader = self.fake_dataloader(10)
            self.val_loader = self.fake_dataloader(5)
        else:            
            dataset = ArcoV2DatasetV2(self.hparams['fn_zarr'],
                                      years=self.hparams['years'],
                                       sequence_length=self.hparams['sequence_length'],
                                       limit=self.hparams['limit'],
                                       device=torch.device(device)
            )

            train_dataset, valid_dataset = dataset.get_train_validation_subset(0.2)
            

            self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
            self.val_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])

    def train_dataloader(self) -> DataLoader:       
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader