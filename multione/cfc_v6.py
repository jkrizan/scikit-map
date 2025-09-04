#%%

from concurrent.futures import ThreadPoolExecutor, as_completed
from os import read
import torch
from torch import nn
from typing import List, Optional, Union, Any
from ncps.torch.lstm import LSTMCell
from tqdm import tqdm
from cfc_cell import CfCCell 
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset
from numpy.typing import NDArray
import copy
import zarr
from datetime import datetime, timedelta
import numpy as np
from utils import n_imag_per_year
from torch.optim.lr_scheduler import LinearLR
from numba import njit, prange
import gc

class ArcoV2DatasetV6(Dataset):
    def __init__(self, 
                 zarr_path, 
                 years, 
                 sequence_length: int, 
                 band: int,
                 limit=None, 
                 percent_pixels: float = 1.0, 
                 device: str = 'cpu', 
                 dtype=torch.float32):
        # import numpy as np; years = np.arange(2020,2024); sequence_length = 12; limit=None; zarr_path  = "/home/josip/arcov2/sample_v1.zarr"
        self.prepared_all_cases = False
        self.percent_pixels = percent_pixels
        self.nthreads = 4
        self.zarr_path = zarr_path
        self.years = years
        self.sequence_length = sequence_length
        self.band = band
        self.n_output_bands = 1
        self.n_features = self.n_output_bands + 2 # modis_ndvi, geom temp

        self.device = device
        self.dtype = dtype

        dates_list=[]
        for y in self.years:
            dates = [datetime(y,1,8)+timedelta(days=16*i) for i in range(n_imag_per_year)]
            dates_list.extend(dates)        
        self.days_from_start = torch.tensor(np.array([(d - datetime(self.years[0],1,1)).days + 1 for d in dates_list]))
        self.dates = np.array(dates_list).astype('datetime64[D]')
        self.ind_months = np.array([d.month for d in self.dates.astype(object)])
        self.ind_doys_cpu = np.arange(len(self.dates)) % n_imag_per_year
        self.ind_doys = torch.tensor(self.ind_doys_cpu)

        del dates_list

        if self.zarr_path is not None:
            dataset: zarr.Group = zarr.open(zarr_path, mode='r') #type: ignore
            tiles = list(dataset.group_keys())
            tiles = sorted(tiles)
            generator = np.random.default_rng(45)
            self.tiles = generator.permutation(tiles) # to get always same order of tiles
            if limit is not None:
                if isinstance(limit, int):
                    self.tiles = self.tiles[:limit]
                elif isinstance(limit, (list, tuple)):
                    self.tiles = self.tiles[limit[0]:limit[1]]

            #self.timeless_data=[None] * len(self.tiles)
            self.data = [None] * len(self.tiles)
            self.pixel_indices = []
            self.ncases = 0
            self.npixels = 0
            #scaler = StandardScaler() 
            with ThreadPoolExecutor(max_workers= self.nthreads) as executor:
                futures=[executor.submit(self._read_tile_from_zarr,self.zarr_path,tile, tj) for tj, tile in enumerate(self.tiles)]
                for future in tqdm(as_completed(futures), total=len(futures), desc='Reading tiles'):
                    tj, tile, tile_data = future.result()    #type: ignore                    
                    # tj, tile, tile_data = self._read_tile_from_zarr(self.zarr_path, self.tiles[1], 1)
                    
                    tile_nts = tile_data[0]
                    # tile_cases = tile_nts.sum()
                    for pj in range(tile_nts.shape[0]):
                        self.pixel_indices.extend([(tj, pj, jts) for jts in range(tile_nts[pj])])
                        self.ncases += tile_nts[pj]

                    data = [torch.tensor(d, dtype=self.dtype) for d in tile_data[1:-1]]
                    data.append(torch.tensor(tile_data[-1])) # all_valid_values as boolean
                    data.append(tile_nts)  # add nts at the end, as int32

                    self.data[tj] = data
                    
                    self.npixels += tile_nts.shape[0]
                                                        
            # if self.read_timeless:
            #     self.n_timeless_features = self.timeless_data[0].shape[-1]            
            
            self.length = self.ncases
            self.subset = np.array(range(self.length))
            # self.data_scaler = scaler
            # self.data_scaler = self.compute_data_scaler()

    def _read_tile_from_zarr(self, zarr_path, tile, tj:int):
        # tile = self.tiles[0]
        dataset: zarr.Group = zarr.open(zarr_path, mode='r') #type: ignore
        group: zarr.Group = dataset[tile]   #type: ignore

        # if self.read_timeless:
        #     covariates: NDArray = group['covariates'][:] #type: ignore
        #     covariates[np.isnan(covariates)] = 0
        lsdata: NDArray = group['lsdata'][self.band,:,:].squeeze()    #type: ignore
        msdata: NDArray = group['modis'][:]      #type: ignore
        gtemp: NDArray = group['geom_temp_doy'][:]  #type: ignore        
        valid_values_mask: NDArray = group['valid_values_mask'][:]  #type: ignore
        gtemp_min = gtemp.min(axis=0)
        gtemp_max = gtemp.max(axis=0)

        del dataset, group

        if self.percent_pixels<1.0:
            npix = lsdata.shape[1]
            nsel = int(npix*self.percent_pixels)
            generator = np.random.default_rng(45)
            inds = generator.permutation(npix)[:nsel]
            lsdata = lsdata[:,inds]
            msdata = msdata[:,inds]
            gtemp = gtemp[:,inds]
            valid_values_mask = valid_values_mask[:,inds]
            # if self.read_timeless:
            #     covariates = covariates[inds,:]

        npixels = lsdata.shape[1]   # number of sampled pixels in one tile
        
        nts = np.empty((npixels,), dtype=np.int32)
        msdata[np.isnan(msdata)] = -1
        #print(lsdata.shape, msdata.shape, gtemp.shape)
        #all_valid_values = np.isfinite(lsdata) & (msdata>=0) & np.isfinite(gtemp[self.ind_doys_cpu])  #type: ignore

        for j in range(npixels):
            valid_values = valid_values_mask[:, j]   # type: ignore
            nvv = valid_values.sum()
            nts[j] = nvv - self.sequence_length

        data=[nts, lsdata, msdata, gtemp, gtemp_min, gtemp_max, valid_values_mask]
        # if self.read_timeless:
        #     data.append(covariates)

        return tj, tile, tuple(data)

    def get_one_case(self, idx: int):
        tile_ind, pix_ind, ts_ind = self.pixel_indices[idx]
        if self.prepared_all_cases:
            return (
                self.all_tiles[tile_ind][0][ts_ind], 
                self.all_tiles[tile_ind][1][ts_ind,:,:], 
                self.all_tiles[tile_ind][2][ts_ind,:],
                self.all_tiles[tile_ind][3][ts_ind,:]
            )

        (lsdata, msdata, gtemp, gtemp_min, gtemp_max, all_valid_values, tile_nts) = self.data[tile_ind]
        # if self.read_timeless:
        #     covariates = self.timeless_data[tile_ind]

        # Data for one pixel        
        #print(all_valid_values.device, pix_ind)
        valid_values = all_valid_values[:, pix_ind]
        #valid_values_cpu = valid_values.cpu()
        j_dates = self.days_from_start[valid_values]
        j_lsdata = lsdata[valid_values, pix_ind]
        j_msdata = msdata[valid_values, pix_ind]
        j_gtemp = gtemp[self.ind_doys[valid_values], pix_ind]
        j_gtemp_min = gtemp_min[pix_ind]
        j_gtemp_max = gtemp_max[pix_ind]
        #nlsdata = j_lsdata.shape[0]

        #Data for timeseries
        x = torch.cat([j_lsdata[ts_ind:ts_ind+self.sequence_length].reshape(-1,1),
                            j_msdata[ts_ind:ts_ind+self.sequence_length].reshape(-1,1),
                            j_gtemp[ts_ind:ts_ind+self.sequence_length].reshape(-1,1)]
                            , dim=1)
        gtemp = torch.cat([j_gtemp_min, j_gtemp_max]).reshape(1,2)

        y = j_lsdata[ts_ind+self.sequence_length]
        timespans = (j_dates[ts_ind + 1: ts_ind+1+self.sequence_length] - j_dates[ts_ind:ts_ind+self.sequence_length])/366
        timespans = timespans.to(self.dtype)

        return (y, x, gtemp, timespans)

    
    def prepare_all_cases(self):
        
        @njit(parallel=True)
        def process_tile(y, x, x_gtemp, timespans_data, 
                         lsdata, msdata, 
                         gtemp, gtemp_min, gtemp_max,
                         all_valid_values, tile_nts,
                         ind_doys, days_from_start, ind_months,
                         sequence_length):
            npix = tile_nts.shape[0]
            ncases = tile_nts.cumsum()            
            for pix_ind in prange(npix):
                # pix_ind=0
                valid_values = all_valid_values[:, pix_ind]                
                j_dates = days_from_start[valid_values]
                j_lsdata = lsdata[valid_values, pix_ind]
                j_msdata = msdata[valid_values, pix_ind]
                j_gtemp = gtemp[ind_doys[valid_values], pix_ind]

                ncase = ncases[pix_ind-1] if pix_ind>0 else 0
                for ts_ind in range(tile_nts[pix_ind]):
                    # ts_ind=0
                    x[ncase, :, 0] = j_lsdata[ts_ind:ts_ind + sequence_length]
                    x[ncase, :, 1] = j_msdata[ts_ind:ts_ind + sequence_length]
                    x[ncase, :, 2] = j_gtemp[ts_ind:ts_ind + sequence_length]

                    x_gtemp[ncase, 0] = gtemp_min[pix_ind]
                    x_gtemp[ncase, 1] = gtemp_max[pix_ind]
                    x_gtemp[ncase, 2] = j_gtemp[ts_ind+sequence_length]
                    y[ncase] = j_lsdata[ts_ind + sequence_length]

                    timespans_data[ncase, :] = (j_dates[ts_ind + 1: ts_ind + 1 + sequence_length] - j_dates[ts_ind:ts_ind + sequence_length]) / 366
                    ncase += 1


        tiles = []
        pixinds = np.array(self.pixel_indices)
        dtype = np.float32
        ttype = torch.float32

        for t in range(len(self.tiles)):
            # t = 0
            (lsdata, msdata, gtemp, gtemp_min, gtemp_max, all_valid_values,tile_nts) = self.data[t]
            ncases = tile_nts.sum()
            y = np.empty((ncases,), dtype=dtype)
            x = np.empty((ncases, self.sequence_length, self.n_features), dtype=dtype)
            x_gtemp = np.empty((ncases, 3), dtype=dtype)
            timespans_data = np.empty((ncases, self.sequence_length), dtype=dtype)            
        
            lsdata = lsdata.to(ttype).numpy()
            msdata = msdata.to(ttype).numpy()
            gtemp = gtemp.to(ttype).numpy()
            all_valid_values = all_valid_values.to(torch.bool).numpy()
            days_from_start = self.days_from_start.to(torch.int16).numpy()
            sequence_length = self.sequence_length
            ind_doys = self.ind_doys.to(torch.int16).numpy()
            gtemp_min = gtemp_min.to(ttype).numpy()
            gtemp_max = gtemp_max.to(ttype).numpy()
        
            process_tile(y, x, x_gtemp, timespans_data, 
                     lsdata, msdata, 
                     gtemp, gtemp_min, gtemp_max,
                     all_valid_values, tile_nts,
                     ind_doys, days_from_start, self.ind_months,
                     sequence_length)

            y = torch.tensor(y, dtype=self.dtype).to(self.device)
            x = torch.tensor(x, dtype=self.dtype).to(self.device)
            x_gtemp = torch.tensor(x_gtemp, dtype=self.dtype).to(self.device)
            #gtemp_data = torch.cat([gtemp_min.unsqueeze(1), gtemp_max.unsqueeze(1)], dim=1).to(self.dtype).to(self.device)
            timespans_data = torch.tensor(timespans_data, dtype=self.dtype).to(self.device)

            tiles.append((y, x, x_gtemp, timespans_data))

            del lsdata, msdata, gtemp, all_valid_values, tile_nts
            torch.cuda.empty_cache()
            gc.collect()

        self.prepared_all_cases = True
        self.all_tiles = tiles
        del self.tiles
        gc.collect()


    def get_train_validation_subset(self, ncases_validation: float):
        indices = np.array(range(self.length))
        rndgen = np.random.default_rng(43)
        rndgen.shuffle(indices)
        ncases_validation = int(ncases_validation * len(indices))
        #train_dataset = copy.copy(self)
        #valid_dataset = copy.copy(self)
        train_subset = Subset(self, indices[ncases_validation:])
        valid_subset = Subset(self, indices[:ncases_validation])
        return train_subset, valid_subset

    def set_subset(self, subset: NDArray):
        self.subset = subset
        self.length = len(subset)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        idx = self.subset[idx]
        return self.get_one_case(idx)

    # def compute_data_scaler(self):
    #      scaler = StandardScaler()                 
    #      for data in self.data:
    #          data_to_scale = np.concatenate([
    #              data[1].reshape((6,-1)).T, 
    #              data[2].reshape((1,-1)).T, 
    #              np.repeat(data[3], len(self.years)).reshape((1,-1)).T], 
    #              axis=1)             
    #          scaler.partial_fit(data_to_scale)
    #      return scaler
    
class CfcModelV6(nn.Module):
    def __init__(self, 
                 input_size:int, 
                 hidden_size: int, 
                 activation: str = 'lecun_tanh', #silu, relu, tanh, gelu, lecun_tanh
                 sequence_length:int = 12, 
                 backbone_layers:list[int]=[128, 64, 32],
                 backbone_dropout: float = 0.1, 
                 output_size:int = 1):

        super(CfcModelV6, self).__init__()

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
        

        # self.fc_timeless = nn.Sequential(
        #     nn.Linear(17, 32),
        #     nn.ReLU(),
        #     #nn.Dropout(backbone_dropout),
        #     nn.Linear(32, 16),
        #     nn.ReLU(),
        #     #nn.Dropout(backbone_dropout),
        #     nn.Linear(16, 8),
        # )
        self.rnn = CfCCell(
                self.input_size,
                self.hidden_size,
                self.mode,
                self.activation,
                self.backbone_layers,
                self.backbone_dropout,
                )
        self.lstm = LSTMCell(self.input_size, self.hidden_size)  # Mixed memory
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size+2*self.output_size, self.hidden_size//2),
            nn.ReLU(),
            nn.Linear(self.hidden_size//2, self.hidden_size//4),
            nn.ReLU(),
            nn.Linear(self.hidden_size//4, self.output_size)
        )

        #print(f"CfcModel_v3: device={self.fc.weight.device}, {self.rnn_sequence[0].ff1.weight.device}")
        self.init_weights()
        

    def forward(self, x, timespans, hx=None):
        # x (batch, 1, seq_len, input_size)
        device = x.device
        dtype = x.dtype
        x = x.squeeze(1)  # (batch, seq_len, input_size)
        timespans = timespans.squeeze(1)  # (batch, seq_len)
        batch_size, seq_len = x.size(0), x.size(1)

        if hx is None:
            h_state = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
            c_state = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
        else:
            h_state, c_state = hx
        
        x_mean = x[:,:,:self.output_size].mean(dim=1).detach()
        x_std = x[:,:,:self.output_size].std(dim=1).detach()
        #timeless = self.fc_timeless(timeless)
        for t in range(seq_len):
            #inputs = torch.concatenate([x[:, t], timeless], dim=1)
            inputs = x[:, t, :]
            
            ts = 1.0 if timespans is None else timespans[:, t].reshape(-1,1) #.squeeze()

            h_state, c_state = self.lstm(x[:,t], (h_state, c_state))
            h_out, h_state = self.rnn(inputs, ts, timeless=None, hx=h_state)

        #merged = torch.cat([h_out, timeless], dim=1)
        #print(f"h_out: {h_out.shape}, x_mean: {x_mean.shape}")
        merged = torch.cat([h_out, x_mean, x_std], dim=1)
        readout = self.fc(merged) #type: ignore
        #hx = (h_state, c_state) #if self.use_mixed else h_state

        return readout #, x.mean(dim=1)[:,:self.output_size] #, hx


    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)
            else:
                torch.nn.init.uniform_(w)



        

class CfcLearnerV6(pl.LightningModule):
    def __init__(self, fn_zarr, years, 
                input_size:int, hidden_size:int, sequence_length:int, band: int,
                output_size:int, backbone_layers, limit:int, 
                activation: str, lr:float=0.01, 
                debug=False, dtype: torch.dtype=torch.float32, device='cuda',
                batch_size: int = 32,
                #criterion: str = None
                ):
        super(CfcLearnerV6, self).__init__()
        self.to(dtype)
        # self.criterion = nn.MSELoss() 
        # if criterion == 'special':
        
        self.criterion = nn.MSELoss() #self.special_criterion
        self.zero = torch.tensor(0.0, dtype=dtype, device=device)
        self.one = torch.tensor(1.0, dtype=dtype, device=device)
        # DONE: Make criterion that weight of error is inversly proportional of difference between 
        # target observed value and mean of previously observed values in timeseries
        # that way model will not try to predict outliers

        self.model = CfcModelV6(input_size=input_size,
                                hidden_size=hidden_size, 
                                sequence_length=sequence_length, 
                                output_size=output_size, 
                                backbone_layers=backbone_layers, 
                                backbone_dropout=0,
                                activation=activation)
        
        self.save_hyperparameters()

    def special_criterion(self, means, predicted, observed):
        # means: (batch, input_size)
        # predicted: (batch, output_size)
        # observed: (batch, output_size)
        # calculate weights
        #weights = torch.maximum(1-(torch.abs(observed - means)**2),self.zero).detach()
        weights = torch.clamp(1 - torch.abs(observed - means), min=0.0, max=1.0).detach()
        if weights.max() == 0:
            weights = self.one
        #weights_sum = torch.sum(weights)
        
        #if weights_sum == torch.inf:
        #    weights_sum = weights.max()

        loss = torch.sum(weights*(predicted - observed)**2) #/weights_sum
        #return nn.MSELoss(reduction='none')(predicted * weights, observed * weights).mean()
        return loss

    def forward(self, x, timespans):
        res = self.model.forward(x, timespans)
        return res

    def training_step(self, batch, batch_idx):
        (y, x, timespans) = batch
        y_hat = self.model(x, timespans)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        #loss = self.criterion(x[:, :, 0].mean(dim=1), y_hat.squeeze(), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=y.shape[0],  sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (y, x, timespans) = batch
        y_hat = self.model(x, timespans)
        loss = self.criterion(y_hat.squeeze(), y.squeeze())
        #loss = self.criterion(x[:, :, 0].mean(dim=1), y_hat.squeeze(), y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=y.shape[0],  sync_dist=True)
        return loss

    def fake_dataloader(self,n:int):
        device = torch.cuda.current_device()
        t = torch.randn(2, device=device)
        #print(f"Fake dataloader: {t.device}")
        return (
            (i,
             (torch.randn(2, self.hparams['output_size'], device=device),
              #torch.randn(2, 1, self.hparams['sequence_length'], self.hparams['input_size'], device=device),
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
        #device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        #print(f'Setup, device = {torch.get_device()}')
        #self.model.transfer_to_device(device)

        #print(f"Setup, poslije: device={self.model.fc.weight.device}, {self.model.rnn_sequence[0].ff1.weight.device}")¸

        # dtype = self.hparams['dtype']
        # if dtype == 'float32':
        #     self.dtype = torch.float32
        # elif dtype=='float16':
        #     self.dtype = torch.float16
        # elif dtype=='bfloat16':
        #     self.dtype = torch.bfloat16

        if self.hparams['debug']:
            #print(f'Hello from "setup", {stage=}, GPU: {torch.cuda.current_device()}')     
            self.train_loader = self.fake_dataloader(10)
            self.val_loader = self.fake_dataloader(5)
        else:            
            #print(f'Setup, {self.hparams}')
            dataset = ArcoV2DatasetV6(self.hparams['fn_zarr'],
                                      years=self.hparams['years'],
                                       sequence_length=self.hparams['sequence_length'],
                                       band = self.hparams['band'],
                                       limit=self.hparams['limit'],
                                       device='cpu' if self.hparams['device']=='cpu' else f'cuda:{torch.cuda.current_device()}',
                                       dtype=self.dtype                                
            )
            print(f'Setup, device in dataset = {dataset[0][0].device}')
            self.dataset = dataset
            #self.data_scaler = dataset.data_scaler
            print(f"Dataset length: {len(dataset)}")
            #self.data_min_max = dataset.data_min_max()

            #print(dataset[0][0].shape, dataset[0][1].shape, dataset[0][2].shape)
            #print(dataset[0][0].dtype, dataset[0][1].dtype, dataset[0][2].dtype)

            train_subset, valid_subset = dataset.get_train_validation_subset(0.2)
            print(f"Train dataset length: {len(train_subset)}")
            print(f"Validation dataset length: {len(valid_subset)}")

            self.train_loader = DataLoader(train_subset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=4, prefetch_factor=4)
            self.val_loader = DataLoader(valid_subset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=4, prefetch_factor=4)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])
        #lr_scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=300)
        #return [optimizer], [lr_scheduler]
        return optimizer

    def train_dataloader(self) -> DataLoader:       
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return self.val_loader
    
#%%
def playground():
    import numpy as np
    fn_zarr = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
    years = np.arange(2020,2024)
    sequence_length = 12
    limit=10
    band=0
    dataset = ArcoV2DatasetV6(fn_zarr, years, sequence_length, band, limit=limit, device='cpu', dtype=torch.bfloat16)
    print(f"Dataset length: {len(dataset)}")
    print(dataset[0][0].shape, dataset[0][1].shape, dataset[0][2].shape)
    print(dataset[0][0].dtype, dataset[0][1].dtype, dataset[0][2].dtype)

    #train_subset, valid_subset = dataset.get_train_validation_subset(0.2)
    #print(f"Train dataset length: {len(train_subset)}")
    #print(f"Validation dataset length: {len(valid_subset)}")

    #train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4, prefetch_factor=4)
    #val_loader = DataLoader(valid_subset, batch_size=32, shuffle=False, num_workers=4, prefetch_factor=4)

    #for batch in train_loader:
    #    (y, x, timespans) = batch
    #    print(x.shape, y.shape, timespans.shape)
    #    break