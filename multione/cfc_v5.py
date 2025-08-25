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


from concurrent.futures import ThreadPoolExecutor, as_completed
from os import read
import torch
from torch import nn
from typing import List, Optional, Union
from ncps.torch.lstm import LSTMCell
from tqdm import tqdm
from cfc_cell import CfCCell #, WiredCfCCell
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from numpy.typing import NDArray
import copy
import zarr
from datetime import datetime, timedelta
import numpy as np
from utils import n_imag_per_year
from torch.optim.lr_scheduler import LinearLR

class ArcoV2DatasetV3(Dataset):
    def __init__(self, zarr_path, years, sequence_length: int, limit=None, read_timeless=False, device=torch.get_default_device()):
        # years = np.arange(2020,2024); sequence_length = 12; limit=None; zarr_path = Path(f"/data/oemc/arcov2/sample_v1.zarr")
        #if zarr_path is None:
        #    zarr_path = fn_zarr
        self.nthreads = 8
        self.zarr_path = zarr_path
        self.years = years
        self.sequence_length = sequence_length
        self.n_output_bands = 7
        self.n_features = self.n_output_bands + 2 # modis_ndvi, geom temp
        self.read_timeless = read_timeless
        self.device = device

        dates_list=[]
        for y in years:
            dates = [datetime(y,1,8)+timedelta(days=16*i) for i in range(n_imag_per_year)]
            dates_list.extend(dates)        
        self.days_from_start = np.array([(d - datetime(years[0],1,1)).days + 1 for d in dates_list])
        self.dates = np.array(dates_list).astype('datetime64[D]')
        self.ind_doys = np.arange(len(self.dates)) % n_imag_per_year
        del dates_list

        if self.zarr_path is not None:
            dataset: zarr.Group = zarr.open(zarr_path, mode='r') #type: ignore
            tiles = list(dataset.group_keys())
            tiles = sorted(tiles)
            generator = np.random.default_rng(43)
            self.tiles = generator.permutation(tiles) # to get always same order of tiles
            if limit is not None:
                if isinstance(limit, int):
                    self.tiles = self.tiles[:limit]
                elif isinstance(limit, (list, tuple)):
                    self.tiles = self.tiles[limit[0]:limit[1]]

            self.timeless_data=[None] * len(self.tiles)
            self.data = [None] * len(self.tiles)
            self.pixel_indices = []
            self.ncases = 0
            self.npixels = 0
            with ThreadPoolExecutor(max_workers= self.nthreads) as executor:
                futures=[executor.submit(self._read_tile_from_zarr,self.zarr_path,tile, tj) for tj, tile in enumerate(self.tiles)]
                for future in tqdm(as_completed(futures), total=len(futures), desc='Reading tiles'):
                    tj, tile, tile_data = future.result()    #type: ignore                    
                    # tj, tile, tile_data = self._read_tile_from_zarr(self.zarr_path, self.tiles[1], 1)
                    
                    tile_nts = tile_data[0]
                    tile_cases = tile_nts.sum()
                    for pj in range(tile_nts.shape[0]):
                        self.pixel_indices.extend([(tj, pj, jts) for jts in range(tile_nts[pj])])
                        self.ncases += tile_nts[pj]

                    self.data[tj]=tile_data[:5]
                    if self.read_timeless:                        
                        self.timeless_data[tj]=tile_data[5] #type: ignore

                    self.npixels += tile_nts.shape[0]
                                                        
            if self.read_timeless:
                self.n_timeless_features = self.timeless_data[0].shape[-1]            
            
            self.length = self.ncases
            self.subset = np.array(range(self.length))

    def _read_tile_from_zarr(self, zarr_path, tile, tj:int):
        # tile = self.tiles[0]
        dataset: zarr.Group = zarr.open(zarr_path, mode='r') #type: ignore
        group: zarr.Group = dataset[tile]   #type: ignore

        if self.read_timeless:
            covariates: NDArray = group['covariates'][:] #type: ignore
            covariates[np.isnan(covariates)] = 0
        lsdata: NDArray = group['lsdata'][:]    #type: ignore
        msdata: NDArray = group['modis'][:]      #type: ignore
        gtemp: NDArray = group['geom_temp_doy'][:]  #type: ignore
        gtemp[np.isnan(gtemp)] = 0

        del dataset, group

        npixels = lsdata.shape[2]   # number of sampled pixels in one tile
        
        nts = np.empty((npixels,), dtype=np.int32)
        all_valid_values = np.isfinite(lsdata).all(axis=0)# & np.isfinite(msdata).all(axis=0) & np.isfinite(gtemp).all(axis=0)
        for j in range(npixels):
            valid_values = all_valid_values[:, j]   # type: ignore
            nvv = valid_values.sum()
            nts[j] = nvv - self.sequence_length
        
        data=[nts, lsdata, msdata, gtemp, all_valid_values]
        if self.read_timeless:
            data.append(covariates)

        return tj, tile, tuple(data)

    def get_one_case(self, idx: int):
        tile_ind, pix_ind, ts_ind = self.pixel_indices[idx]
        (nts, lsdata, msdata, gtemp, all_valid_values) = self.data[tile_ind]
        if self.read_timeless:
            covariates = self.timeless_data[tile_ind]

        # Data for one pixel        
        valid_values = all_valid_values[:, pix_ind]
        j_dates = self.days_from_start[valid_values]
        j_lsdata = lsdata[:, valid_values, pix_ind]
        j_msdata = msdata[valid_values, pix_ind]
        j_gtemp = gtemp[self.ind_doys[valid_values], pix_ind]
        nlsdata = j_lsdata.shape[0]

        #Data for timeseries
        x = np.concatenate([j_lsdata[:, ts_ind:ts_ind+self.sequence_length].T, 
                            j_msdata[ts_ind:ts_ind+self.sequence_length].reshape(-1,1)/10000,
                            j_gtemp[ts_ind:ts_ind+self.sequence_length].reshape(-1,1)/100]
                            , axis=1)
        y = j_lsdata[:,ts_ind+self.sequence_length]  
        timespans = (j_dates[ts_ind + 1: ts_ind+1+self.sequence_length] - j_dates[ts_ind:ts_ind+self.sequence_length])/36

        x[np.isnan(x)] = 0

        #with torch.device(self.device):
        res = [torch.tensor(y), torch.tensor(x), torch.tensor(timespans,dtype=torch.float32)]
        if self.read_timeless:
            x_timeless = covariates[:,pix_ind]  # type: ignore
            res.append(torch.tensor(x_timeless,dtype=torch.float32))

        return tuple(res)

    def get_one_pixel(self, tile:str|int, pixel_ind:int):
        tile_ind = self.tiles.index(tile) if isinstance(tile, str) else tile
        # TODO: 
        # return batch of all valid timeseries in this pixel, and dates for y, and y, and timespans
        #return self.data[tile_ind][pixel_ind]

    def get_train_validation_subset(self, ncases_validation: float):
        indices = np.array(range(self.length))
        np.random.shuffle(indices)
        ncases_validation = int(ncases_validation * len(indices))
        train_dataset = copy.copy(self)
        valid_dataset = copy.copy(self)
        train_dataset.set_subset(indices[ncases_validation:])
        valid_dataset.set_subset(indices[:ncases_validation])
        return train_dataset, valid_dataset

    def set_subset(self, subset: NDArray):
        self.subset = subset
        self.length = len(subset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        idx = self.subset[idx]
        return self.get_one_case(idx)
    
class CfcModel_v5(nn.Module):
    def __init__(self, 
                 input_size:int, 
                 hidden_size: int, 
                 activation: str = 'lecun_tanh', #silu, relu, tanh, gelu, lecun_tanh
                 sequence_length:int = 12, 
                 backbone_layers:list[int]=[128, 64, 32],
                 backbone_dropout: float = 0.1, 
                 output_size:int = 7):

        super(CfcModel_v5, self).__init__()

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
        
        # self.rnn_sequence = nn.ModuleList( [ 
        #     CfCCell(
        #         self.input_size,
        #         self.hidden_size,
        #         self.mode,
        #         self.activation,
        #         self.backbone_layers,
        #         self.backbone_dropout,
        #         )
        #         for _ in range(self.sequence_length)
        #     ])
        self.fc_timeless = nn.Sequential(
            nn.Linear(17, 32),
            nn.ReLU(),
            #nn.Dropout(backbone_dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            #nn.Dropout(backbone_dropout),
            nn.Linear(16, 8),
        )
        self.rnn = CfCCell(
                self.input_size+8,
                self.hidden_size,
                self.mode,
                self.activation,
                self.backbone_layers,
                self.backbone_dropout,
                )
        self.lstm = LSTMCell(self.input_size+8, self.hidden_size)  # Mixed memory
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, self.output_size)
        )

        #print(f"CfcModel_v3: device={self.fc.weight.device}, {self.rnn_sequence[0].ff1.weight.device}")
        self.init_weights()
        
    # def transfer_to_device(self, device):
    #     self.fc = self.fc.to(device)
    #     self.lstm = self.lstm.to(device)
    #     self.rnn_sequence = [cell.to(device) for cell in self.rnn_sequence]

    def forward(self, x, timespans, timeless, hx=None):
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
        
        timeless = self.fc_timeless(timeless)
        for t in range(seq_len):
            inputs = torch.concatenate([x[:, t], timeless], dim=1)
            
            ts = 1.0 if timespans is None else timespans[:, t].reshape(-1,1) #.squeeze()

            h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out, h_state = self.rnn(inputs, ts, timeless, h_state)

        readout = self.fc(h_out) #type: ignore
        #hx = (h_state, c_state) #if self.use_mixed else h_state

        return readout #, hx


    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)
            else:
                torch.nn.init.uniform_(w)



        

class CfcLearner_v5(pl.LightningModule):
    def __init__(self, fn_zarr, years, input_size:int, hidden_size:int, sequence_length:int, output_size:int, backbone_layers, limit:int, activation: str, lr:float=0.01, debug=False):
        super(CfcLearner_v5, self).__init__()
        self.criterion = nn.MSELoss()

        self.model = CfcModel_v5(input_size=input_size,
                                hidden_size=hidden_size, 
                                sequence_length=sequence_length, 
                                output_size=output_size, 
                                backbone_layers=backbone_layers, 
                                backbone_dropout=0.1,
                                activation=activation)

        #self.model = SimpleModel_v2(output_size)
        #_ = self.model.forward(torch.randn(1, 1, sequence_length, input_size), torch.randn(1, 1, sequence_length)) #init Lazy ones   #batch_size, channel_size, height, width
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x, timespans, timeless):
        return self.model.forward(x, timespans, timeless)

    def training_step(self, batch, batch_idx):
        (y, x, timespans, timeless) = batch
        y_hat = self(x, timespans, timeless)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=y.shape[0],  sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (y, x, timespans, timeless) = batch
        y_hat = self(x, timespans, timeless)
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
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        #print(f'Setup, device={device}, {torch.device(device)}')
        #self.model.transfer_to_device(device)

        #print(f"Setup, poslije: device={self.model.fc.weight.device}, {self.model.rnn_sequence[0].ff1.weight.device}")

        if self.hparams['debug']:
            #print(f'Hello from "setup", {stage=}, GPU: {torch.cuda.current_device()}')     
            self.train_loader = self.fake_dataloader(10)
            self.val_loader = self.fake_dataloader(5)
        else:            
            #print(f'Setup, {self.hparams}')
            dataset = ArcoV2DatasetV3(self.hparams['fn_zarr'],
                                      years=self.hparams['years'],
                                       sequence_length=self.hparams['sequence_length'],
                                       limit=self.hparams['limit'],
                                       read_timeless=True,
                                       device=torch.device(device)
            )
            print(f"Dataset length: {len(dataset)}")
            #print(dataset[0][0].shape, dataset[0][1].shape, dataset[0][2].shape)
            #print(dataset[0][0].dtype, dataset[0][1].dtype, dataset[0][2].dtype)

            train_dataset, valid_dataset = dataset.get_train_validation_subset(0.2)
            print(f"Train dataset length: {len(train_dataset)}")
            print(f"Validation dataset length: {len(valid_dataset)}")

            self.train_loader = DataLoader(train_dataset, batch_size=8192, shuffle=True, num_workers=8)
            self.val_loader = DataLoader(valid_dataset, batch_size=8192, shuffle=False, num_workers=8)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])
        lr_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self) -> DataLoader:       
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader