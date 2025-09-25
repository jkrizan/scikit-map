#%%

from concurrent.futures import ThreadPoolExecutor, as_completed
from os import read
from numpy.ma import indices
import torch
from torch import Tensor, nn
from typing import List, Optional, Union, Any
from ncps.torch.lstm import LSTMCell
from tqdm import tqdm
from cfc_cell import CfCCell 
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset
from numpy.typing import ArrayLike, NDArray
import copy
import zarr
from datetime import datetime, timedelta
import numpy as np
from utils import n_imag_per_year
from torch.optim.lr_scheduler import LinearLR
from numba import njit, prange
import gc
import copy
import time

class MemoryDataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, indexes: ArrayLike | None):
        self.dataset = dataset
        self.batch_size = batch_size
        if indexes is None:
            indexes = np.array(range(len(self.dataset))) # type: ignore
        self.indexes = indexes
        self.max_idx = len(self.indexes)//self.batch_size   # type: ignore
        self.batch_idx = 0
        
        self.rng = np.random.default_rng()

    def __iter__(self):
        return self
    
    def __len__(self):
        return self.max_idx

    def __next__(self):
        return self.get_batch()

    def shuffler(self):   
        indices = self.indexes.cpu().numpy() if isinstance(self.indexes, torch.Tensor) else self.indexes     
        self.rng.shuffle(indices)
        self.indexes = torch.tensor(indices, device=self.indexes.device) if isinstance(self.indexes, torch.Tensor) else indices

    def get_batch(self):
        if self.batch_idx >= self.max_idx:
            self.batch_idx = 0
            self.shuffler()
            raise StopIteration
        else:
            start = self.batch_idx * self.batch_size
            end = start + self.batch_size
            batch_indexes = self.indexes[start:end] # type: ignore
            self.batch_idx += 1

            return self.dataset.get_cases(batch_indexes)    # type: ignore
            #return tuple([torch.stack([self.dataset[i][j] for i in batch_indexes], dim=0) for j in range(len(self.dataset[0]))])
            #batch = [self.dataset[i] for i in batch_indexes]
            #return tuple([torch.stack([b[j] for b in batch], dim=0) for j in range(len(batch[0]))])
        
        

class ArcoV2DatasetV8(Dataset):
    def clone_to(self, device: str):
        self.device = device
        
        if self.prepared_all_cases:
            new_self = copy.copy(self)
            new_self.all_y = self.all_y.to(device)
            new_self.all_x = self.all_x.to(device)
            new_self.all_x_gtemp = self.all_x_gtemp.to(device)
            new_self.all_timespans = self.all_timespans.to(device)
            
            return new_self
        else:
            raise Exception("Data not prepared, cannot clone to device")

    def __init__(self, 
                 zarr_path, 
                 years, 
                 sequence_length: int, 
                 bands: list[int],
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
        self.bands = bands
        self.n_output_bands = len(self.bands)
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
            generator = np.random.default_rng(46)
            self.tiles = generator.permutation(tiles) # to get always same order of tiles
            if limit is not None:
                if isinstance(limit, int):
                    self.tiles = self.tiles[:limit]
                elif isinstance(limit, (list, tuple)):
                    self.tiles = self.tiles[limit[0]:limit[1]]

            self.data: List = [None] * len(self.tiles)
            self.pixel_indices_list: List = []
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
                        self.pixel_indices_list.extend([(tj, pj, jts) for jts in range(tile_nts[pj] - 1)])
                        self.ncases += tile_nts[pj] - 1
                        

                    data: List = [torch.tensor(d, dtype=self.dtype) for d in tile_data[1:-1]]
                    data.append(tile_data[-1].astype(bool)) # all_valid_values as boolean
                    data.append(tile_nts)  # add nts at the end, as int32

                    self.data[tj] = data
                    
                    self.npixels += tile_nts.shape[0]
                                                        
            # if self.read_timeless:
            #     self.n_timeless_features = self.timeless_data[0].shape[-1]            
            
            self.length = self.ncases
            self.subset = np.array(range(self.length))
            self.pixel_indices:np.ndarray = np.array(self.pixel_indices_list, dtype=np.int32) # (tile_ind, pixel_ind, ts_ind)

            # self.data_scaler = scaler
            # self.data_scaler = self.compute_data_scaler()

    def _read_tile_from_zarr(self, zarr_path, tile, tj:int):
        # tile = self.tiles[0]
        dataset: zarr.Group = zarr.open(zarr_path, mode='r') #type: ignore
        group: zarr.Group = dataset[tile]   #type: ignore

        # if self.read_timeless:
        #     covariates: NDArray = group['covariates'][:] #type: ignore
        #     covariates[np.isnan(covariates)] = 0
        lsdata: NDArray = group['lsdata'][self.bands,:,:]    #type: ignore
        msdata: NDArray = group['modis'][:]      #type: ignore
        gtemp: NDArray = group['geom_temp_doy'][:]  #type: ignore        
        valid_values_mask: NDArray = group['valid_values_mask'][:]  #type: ignore
        gtemp_min = gtemp.min(axis=0)
        gtemp_max = gtemp.max(axis=0)

        del dataset, group

        npixels = lsdata.shape[2]   # number of sampled pixels in one tile
        if self.percent_pixels<1.0:
            npixels = int(npixels*self.percent_pixels)
            generator = np.random.default_rng(46)
            inds = generator.permutation(npixels)[:npixels]
            lsdata = lsdata[:,:,inds]
            msdata = msdata[:,inds]
            gtemp = gtemp[:,inds]
            valid_values_mask = valid_values_mask[:,inds]

        nts = np.empty((npixels,), dtype=np.int32)
        msdata[np.isnan(msdata)] = -1

        for j in range(npixels):
            valid_values = valid_values_mask[:, j]   # type: ignore
            nvv = valid_values.sum()
            nts[j] = nvv - self.sequence_length

        data=[nts, lsdata, msdata, gtemp, gtemp_min, gtemp_max, valid_values_mask]

        return tj, tile, tuple(data)

    def get_cases(self, indices: ArrayLike) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        y = self.all_y[indices]
        x = self.all_x[indices]
        x_gtemp = self.all_x_gtemp[indices]
        timespans = self.all_timespans[indices]
        return (y, x, x_gtemp, timespans)

    def get_one_case(self, idx: int):
        
        if self.prepared_all_cases:
            return (
                #torch.nan_to_num(self.all_tiles[tile_ind][0][ts_ind],0,0,0),
                #torch.nan_to_num(self.all_tiles[tile_ind][1][ts_ind,:,:],0,0,0), 
                #torch.nan_to_num(self.all_tiles[tile_ind][2][ts_ind,:],0,0,0),
                #torch.nan_to_num(self.all_tiles[tile_ind][3][ts_ind,:],0,0,0)
                # self.all_tiles[tile_ind][0][ts_ind],
                # self.all_tiles[tile_ind][1][ts_ind,:,:], 
                # self.all_tiles[tile_ind][2][ts_ind,:],
                # self.all_tiles[tile_ind][3][ts_ind,:]
                self.all_y[idx], self.all_x[idx], self.all_x_gtemp[idx], self.all_timespans[idx]
            )
        else:
            raise Exception("Data not prepared, cannot get one case")
        
        tile_ind, pix_ind, ts_ind = self.pixel_indices[idx]
        (lsdata, msdata, gtemp, gtemp_min, gtemp_max, all_valid_values, tile_nts) = self.data[tile_ind]
        # if self.read_timeless:
        #     covariates = self.timeless_data[tile_ind]

        # Data for one pixel        
        valid_values = all_valid_values[:, pix_ind]        
        j_dates = self.days_from_start[valid_values]
        j_lsdata = lsdata[:, valid_values, pix_ind]
        j_msdata = msdata[valid_values, pix_ind]
        j_gtemp = gtemp[self.ind_doys[valid_values], pix_ind]
        j_gtemp_min = gtemp_min[pix_ind]
        j_gtemp_max = gtemp_max[pix_ind]
        #nlsdata = j_lsdata.shape[0]

        #Data for timeseries
        x = torch.cat([j_lsdata[:,ts_ind:ts_ind+self.sequence_length].T,
                            j_msdata[ts_ind:ts_ind+self.sequence_length].reshape(-1, 1),
                            j_gtemp[ts_ind:ts_ind+self.sequence_length].reshape(-1,1)]
                            , dim=1)
        
        gtemp = torch.empty((3,), dtype=self.dtype)
        gtemp[0] = j_gtemp_min
        gtemp[1] = j_gtemp_max
        gtemp[2] = j_gtemp[ts_ind+self.sequence_length]

        y = j_lsdata[:,ts_ind+self.sequence_length].to(self.dtype) 
        timespans = (j_dates[ts_ind + 1: ts_ind+1+self.sequence_length] - j_dates[ts_ind:ts_ind+self.sequence_length])/366
        timespans = timespans.to(self.dtype)

        return (y, x, gtemp, timespans)

    
    def prepare_all_cases(self):
        
        @njit(parallel=True)
        def process_tile(y, x, x_gtemp, timespans_data, pixel_indices,
                         lsdata, msdata,
                         gtemp, gtemp_min, gtemp_max,
                         all_valid_values, tile_nts,
                         ind_doys, days_from_start,
                         sequence_length) -> None:
            npix = tile_nts.shape[0]
            ncases = (tile_nts - 1).cumsum()
            nbands = lsdata.shape[0]
            #ncase = 0
            for pix_ind in prange(npix):
                # pix_ind=0
                valid_values = all_valid_values[:, pix_ind]                
                j_dates = days_from_start[valid_values]
                j_lsdata = lsdata[:,valid_values, pix_ind]
                j_msdata = msdata[valid_values, pix_ind]
                j_gtemp = gtemp[ind_doys[valid_values], pix_ind]

                ncase = ncases[pix_ind-1] if pix_ind>0 else 0
                for ts_ind in range(1, tile_nts[pix_ind]):
                    # ts_ind=0
                    for band_ind in range(nbands):
                        x[ncase, :, band_ind] = j_lsdata[band_ind, ts_ind:ts_ind + sequence_length]                    
                    x[ncase, :, nbands] = j_msdata[ts_ind:ts_ind + sequence_length]
                    x[ncase, :, nbands + 1] = j_gtemp[ts_ind:ts_ind + sequence_length]

                    x_gtemp[ncase, 0] = gtemp_min[pix_ind]
                    x_gtemp[ncase, 1] = gtemp_max[pix_ind]
                    x_gtemp[ncase, 2] = j_gtemp[ts_ind+sequence_length]
                    y[ncase, :, 0] = j_lsdata[:, ts_ind + sequence_length]
                    y[ncase, :, 1] = j_lsdata[:, ts_ind - 1]

                    timespans_data[ncase, :] = (j_dates[ts_ind: ts_ind + 1 + sequence_length] - j_dates[ts_ind - 1:ts_ind + sequence_length]) / 366
                    pixel_indices[ncase] = pix_ind
                    ncase += 1
                
        tiles = []

        dtype = np.float32
        ttype = torch.float32

        last_ind = 0
        for t in tqdm(range(len(self.tiles)), desc='Preparing tiles'):
            # t = 0
            (lsdata, msdata, gtemp, gtemp_min, gtemp_max, all_valid_values,tile_nts) = self.data[t]
            ncases = (tile_nts - 1).sum()
            y = np.empty((ncases,self.n_output_bands, 2), dtype=dtype)
            x = np.empty((ncases, self.sequence_length, self.n_features), dtype=dtype)
            x_gtemp = np.empty((ncases, 3), dtype=dtype)
            timespans_data = np.empty((ncases, self.sequence_length + 1), dtype=dtype)
            pixel_indices = np.empty((ncases,), dtype=np.int32)

            lsdata = lsdata.to(ttype).numpy()
            msdata = msdata.to(ttype).numpy()
            gtemp = gtemp.to(ttype).numpy()
            #all_valid_values = all_valid_values.to(torch.bool).numpy()
            days_from_start = self.days_from_start.to(torch.int16).numpy()
            sequence_length = self.sequence_length
            ind_doys = self.ind_doys.to(torch.int16).numpy()
            gtemp_min = gtemp_min.to(ttype).numpy()
            gtemp_max = gtemp_max.to(ttype).numpy()

            process_tile(y, x, x_gtemp, timespans_data, pixel_indices,
                     lsdata, msdata,
                     gtemp, gtemp_min, gtemp_max,
                     all_valid_values, tile_nts,
                     ind_doys, days_from_start,
                     sequence_length)

            y = torch.tensor(y, dtype=self.dtype).to(self.device)
            x = torch.tensor(x, dtype=self.dtype).to(self.device)
            x_gtemp = torch.tensor(x_gtemp, dtype=self.dtype).to(self.device)            
            timespans_data = torch.tensor(timespans_data, dtype=self.dtype).to(self.device)

            tiles.append((y, x, x_gtemp, timespans_data))

            del lsdata, msdata, gtemp, all_valid_values, tile_nts
            self.data[t] = None
            #torch.cuda.empty_cache()
            gc.collect()

            ncases = y.shape[0]
            self.pixel_indices[last_ind:last_ind+ncases,0] = t
            self.pixel_indices[last_ind:last_ind+ncases,1] = pixel_indices
            self.pixel_indices[last_ind:last_ind+ncases,2] = np.arange(ncases)
            last_ind += ncases
            

        self.all_y = torch.cat([tiles[t][0] for t in range(len(tiles))], dim=0)
        self.all_x = torch.cat([tiles[t][1] for t in range(len(tiles))], dim=0)
        self.all_x_gtemp = torch.cat([tiles[t][2] for t in range(len(tiles))], dim=0)
        self.all_timespans = torch.cat([tiles[t][3] for t in range(len(tiles))], dim=0) 

        self.prepared_all_cases = True
        #self.all_tiles = tiles
        del self.data
        gc.collect()

    def get_one_pixel_timeseries(self, tile_ind: int, pix_ind: int):
        # tile_ind, pix_ind = 3,10
        if self.prepared_all_cases:
            raise Exception("Data already prepared, cannot get one pixel timeseries")
        
        (lsdata, msdata, gtemp, gtemp_min, gtemp_max, all_valid_values, tile_nts) = self.data[tile_ind]
        valid_values = np.nonzero(all_valid_values[:, pix_ind])[0]
        
        first_fwd_pos = valid_values[11] + 1
        n_dates = self.days_from_start.shape[0]
        xx = []; timespans=[]; directions=[]
        for ind in range(0, n_dates):
            # ind = 0
            # ind is the index of date to predict
            # need to find 12 valid values
            if ind < first_fwd_pos:
                # need to prepare for bwd direction
                directions.append(1)
                valid_inds = valid_values[valid_values > ind][:12]
                y_dfs = self.days_from_start[ind]
                x_dfs = self.days_from_start[valid_inds]
                ts = np.r_[x_dfs[0]-y_dfs, (x_dfs[1:] - x_dfs[:-1]), 16]            
            else:
                # fwd directions
                directions.append(0)
                valid_inds = valid_values[valid_values < ind][-12:]
                y_dfs = self.days_from_start[ind]
                x_dfs = self.days_from_start[valid_inds]
                ts = np.r_[16, (x_dfs[1:] - x_dfs[:-1]), y_dfs - x_dfs[-1]]
                
            x_lsdata = lsdata[:,valid_inds, pix_ind]
            x_msdata = msdata[valid_inds, pix_ind].reshape(1,-1)
            x_gtemp = gtemp[self.ind_doys[valid_inds], pix_ind].reshape(1,-1)
            x = torch.cat((x_lsdata, x_msdata, x_gtemp), dim=0)            

            xx.append(x)
            timespans.append(ts)

        directions = np.column_stack(directions).astype(np.int8).squeeze()

        x = torch.stack(xx, dim=0).permute(0,2,1).to(self.dtype)  # (n_dates, seq_len, n_features)
        y = lsdata[:, :, pix_ind].T

        timespans = np.column_stack(timespans).T/366
        timespans = torch.tensor(timespans, dtype=self.dtype)

        timeless = torch.empty((n_dates,3), dtype=self.dtype)
        timeless[:,0] = gtemp_min[pix_ind]
        timeless[:,1] = gtemp_max[pix_ind]
        timeless[:,2] = gtemp[self.ind_doys, pix_ind].to(self.dtype)
        
        return (directions, y, x, timeless, timespans, self.dates, valid_values)

    def get_one_pixel_timeseries_v2(self, tile_ind: int, pix_ind: int):
        if not self.prepared_all_cases:
            self.prepare_all_cases()
        

    def get_train_validation_subset(self, ncases_validation: float):
        indices = np.array(range(self.length))
        rndgen = np.random.default_rng(46)
        rndgen.shuffle(indices)
        ncases_validation = int(ncases_validation * len(indices))
        #train_dataset = copy.copy(self)
        #valid_dataset = copy.copy(self)
        train_subset = Subset(self, indices[ncases_validation:])
        valid_subset = Subset(self, indices[:ncases_validation])
        return train_subset, valid_subset
    
    def get_nontraining_subset(self, ncases_validation: float, ncases: int):
        # get subset for testing, not used in training or validation
        indices = np.array(range(self.length))
        rndgen = np.random.default_rng(46)
        rndgen.shuffle(indices)
        ncases_validation = int(ncases_validation * len(indices))
        ncases = min(ncases, ncases_validation)
        subset = Subset(self, indices[:ncases_validation][:ncases])
        return subset

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
class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)
    
class CfcModelV8(nn.Module):
    def __init__(self, 
                 input_size:int, 
                 timeless_input_size: int,
                 hidden_size: int, 
                 activation: str = 'lecun_tanh', #silu, relu, tanh, gelu, lecun_tanh
                 sequence_length:int = 12, 
                 backbone_layers:list[int]=[128, 64, 32],
                 backbone_dropout: float = 0.1, 
                 output_size:int = 1,
                 predict_forward: bool = True,
                 predict_backward: bool = True):

        super(CfcModelV8, self).__init__()

        self.input_size = input_size
        self.timeless_input_size = timeless_input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.output_size = output_size
        self.activation=activation
        self.predict_forward = predict_forward
        self.predict_backward = predict_backward

        self.mode='default'

        self.rnn = CfCCell(
                self.input_size+self.timeless_input_size,
                self.hidden_size,
                self.mode,
                self.activation,
                self.backbone_layers,
                self.backbone_dropout,
                )
        self.lstm_fw = LSTMCell(self.input_size+self.timeless_input_size, self.hidden_size)  # Mixed memory
        self.lstm_bw = LSTMCell(self.input_size+self.timeless_input_size, self.hidden_size)  # Mixed memory
        fc_size = 2*self.hidden_size + 2*self.output_size #
        self.fc = nn.Sequential(
            nn.Linear(fc_size, fc_size//2),
            LeCun(),
            nn.Linear(fc_size//2, fc_size//4),
            LeCun(),
            nn.Linear(fc_size//4, fc_size//8),
            LeCun(),
            nn.Linear(fc_size//8, self.output_size * 2)
        )

        #print(f"CfcModel_v3: device={self.fc.weight.device}, {self.rnn_sequence[0].ff1.weight.device}")
        self.init_weights()

    # def inference_mode(self, x, timeless, timespans, directions):
    #     # directions = 
    #     #   - 1 .. backward, 
    #     #   + 1 .. forward

    #     ind = (directions == -1)
    #     if ind.sum() > 0:
    #         self.predict_forward = True
    #         y_bwd = self(x[ind], timeless[ind], timespans[ind])
    #     else:
    #         y_bwd = None
        
    #     ind = (directions == 1)
    #     if ind.sum() > 0:
    #         self.predict_backward = False
    #         y_fwd = self(x[ind], timeless[ind], timespans[ind])

    def forward(self, x, timeless, timespans):
        # x (batch, 1, seq_len, input_size)
        device = x.device
        dtype = x.dtype
        #x = x.squeeze(1)  # (batch, seq_len, input_size)
        #timespans = timespans.squeeze(1)  # (batch, seq_len)
        batch_size, seq_len = x.size(0), x.size(1)        
        
        x_mean = x[:,:,:self.output_size].mean(dim=1).detach()
        x_std = x[:,:,:self.output_size].std(dim=1).detach()        

        # forward pass
        if self.predict_forward:
        # if hx is None:
            h_state = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
            c_state = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
            # else:
            #     h_state, c_state = hx
            for t in range(seq_len):            
                inputs = torch.cat((x[:, t, :].squeeze(1), timeless), dim=1)
                
                ts = timespans[:, t + 1].reshape(-1,1)

                h_state, c_state = self.lstm_fw(inputs, (h_state, c_state))
                h_out_fw, h_state = self.rnn(inputs, ts, hx=h_state)
        else:
            h_out_fw = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)

        
        # backward pass
        if self.predict_backward:
            #if hx is None:
            h_state = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
            c_state = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
            #else:
            #    h_state, c_state = hx
            for t in reversed(range(seq_len)):
                inputs = torch.cat((x[:, t, :].squeeze(1), timeless), dim=1)            

                ts = timespans[:, t].reshape(-1,1)

                h_state, c_state = self.lstm_bw(inputs, (h_state, c_state))
                h_out_bw, h_state = self.rnn(inputs, ts, hx=h_state)
            else:
                h_out_bw = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
        
        merged = torch.cat([h_out_fw, h_out_bw, x_mean, x_std], dim=1)  #type: ignore
        readout = self.fc(merged) #type: ignore
        #hx = (h_state, c_state) #if self.use_mixed else h_state

        return readout.reshape(-1, self.output_size, 2)


    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)
            else:
                torch.nn.init.uniform_(w)



        

class CfcLearnerV8(pl.LightningModule):
    def __init__(self, fn_zarr, years, 
                input_size:int, hidden_size:int, 
                sequence_length:int, 
                timeless_input_size: int,
                bands: list[int],
                #output_size:int, 
                backbone_layers, 
                limit:int | None, 
                percent_pixels: float,
                activation: str, 
                lr:float=0.01, 
                debug=False, 
                dtype: torch.dtype=torch.float32, 
                device='cuda',
                batch_size: int=4096,
                optimizer = None,                
                ):
        super(CfcLearnerV8, self).__init__()
        self.to(dtype)
        # self.criterion = nn.MSELoss() 
        # if criterion == 'special':
        
        self.optimizer = optimizer
        self.criterion = nn.MSELoss() #self.special_criterion
        self.zero = torch.tensor(0.0, dtype=dtype, device=device)
        self.one = torch.tensor(1.0, dtype=dtype, device=device)
        # DONE: Make criterion that weight of error is inversly proportional of difference between 
        # target observed value and mean of previously observed values in timeseries
        # that way model will not try to predict outliers
        output_size = len(bands)

        self.model = CfcModelV8(input_size=input_size,
                                hidden_size=hidden_size, 
                                timeless_input_size=timeless_input_size,
                                sequence_length=sequence_length, 
                                output_size=output_size, 
                                backbone_layers=backbone_layers, 
                                backbone_dropout=0.0,
                                activation=activation)
        
        #self.model = self.model.to(dtype=dtype)
        self.save_hyperparameters(ignore=['dtype'])

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

    def forward(self, x, timeless, timespans):
        res = self.model.forward(x, timeless, timespans)
        return res

    def training_step(self, batch, batch_idx):
        (y, x, timeless, timespans) = batch
        y_hat = self.model(x, timeless, timespans)
        loss = self.criterion(y_hat, y)
        #loss = self.criterion(x[:, :, 0].mean(dim=1), y_hat.squeeze(), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=y.shape[0],  sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        (y, x, timeless, timespans) = batch
        y_hat = self.model(x, timeless, timespans)
        loss = self.criterion(y_hat, y)
        #loss = self.criterion(x[:, :, 0].mean(dim=1), y_hat.squeeze(), y)
        #self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=y.shape[0],  sync_dist=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=y.shape[0],  sync_dist=True)
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

        if self.hparams['debug']:
            #print(f'Hello from "setup", {stage=}, GPU: {torch.cuda.current_device()}')     
            self.train_loader = self.fake_dataloader(10)
            self.val_loader = self.fake_dataloader(5)
        else:            
            #print(f'Setup, {self.hparams}')
            dataset = ArcoV2DatasetV8( self.hparams['fn_zarr'],
                                       years=self.hparams['years'],
                                       sequence_length=self.hparams['sequence_length'],
                                       bands=self.hparams['bands'],
                                       limit=self.hparams['limit'],
                                       percent_pixels=self.hparams['percent_pixels'] if 'percent_pixels' in self.hparams else 1.0,
                                       device='cpu' if self.hparams['device']=='cpu' else f'cuda:{torch.cuda.current_device()}',
                                       dtype=self.dtype                                
            )
            print("Preparing dataset ...")
            dataset.prepare_all_cases()
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

            if self.hparams['device'] != 'cpu':
                self.train_loader = DataLoader(train_subset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=0) #, prefetch_factor=2)
                self.val_loader = DataLoader(valid_subset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=0) #, prefetch_factor=2)
            else:
                self.train_loader = DataLoader(train_subset, batch_size=self.hparams['batch_size'], persistent_workers=True,
                                               shuffle=True, num_workers=4, prefetch_factor=4)
                self.val_loader = DataLoader(valid_subset, batch_size=self.hparams['batch_size'], persistent_workers=True,
                                             shuffle=False, num_workers=4, prefetch_factor=4)

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'], eps=1e-7)
        ## !!!! eps=1e-7 is important for fp16 training, otherwise it can diverge and loss can be NAN !!!!!!

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
    fn_zarr = "/home/josip/arcov2/sample_v6.zarr"
    years = np.arange(2000,2024)
    sequence_length = 12
    limit=10
    bands=[0, 1, 2, 3, 4, 5] # 
    percent_pixels = 0.1
    ds = ArcoV2DatasetV8(fn_zarr, years, sequence_length, bands, 
                         limit=limit, percent_pixels=percent_pixels, 
                         device='cpu', dtype=torch.float32)
    print(f"Dataset length: {len(ds)}")

    #y1, x1, gt1, ts1 = ds[0]

    ds.prepare_all_cases()
    y, x, tl, ts = ds[0]

    ts, ls = ds.get_train_validation_subset(0.2)
    dl = MemoryDataLoader(ds, batch_size=4096, indexes=ts.indices)

    for i in tqdm(range(len(ds))):
        y, x, gt, ts = ds[i]
        if torch.allclose(y, y1): # or not torch.allclose(x, x1) or not torch.allclose(gt, gt1) or not torch.allclose(ts, ts1):
            print(f"Isti su za {i}")
            break

    (y_obs, x, timeless, timespans, prd_dates, y_dates, prd_dates, valid_values_x, valid_values_y) = ds.get_one_pixel_timeseries(3,10)

    #train_subset, valid_subset = dataset.get_train_validation_subset(0.2)
    #print(f"Train dataset length: {len(train_subset)}")
    #print(f"Validation dataset length: {len(valid_subset)}")

    #train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4, prefetch_factor=4)
    #val_loader = DataLoader(valid_subset, batch_size=32, shuffle=False, num_workers=4, prefetch_factor=4)

    #for batch in train_loader:
    #    (y, x, timespans) = batch
    #    print(x.shape, y.shape, timespans.shape)
    #    break


def profiler_example():
    from torch.profiler import profile, record_function, ProfilerActivity    
    fn_zarr = "/home/josip/arcov2/sample_v6.zarr"
    years = np.arange(2020,2024)
    sequence_length = 12
    limit=5
    bands=[0, 1, 2, 3, 4, 5] # 
    percent_pixels = 0.1
    batch_size=2048*2
    hidden_size=128
    backbone_layers=[128, 64, 32]
    activation='silu'
    lr=0.001
    dtype=torch.float32
    device='cuda'
    gpu_i = torch.cuda.current_device()
    criterion = nn.MSELoss()
    print(f"Profiler example, GPU: {gpu_i}, device name: {torch.cuda.get_device_name(gpu_i)}")

    
    # Your training loop here
    ds = ArcoV2DatasetV7( fn_zarr,
                    years=years,
                    sequence_length=sequence_length,
                    bands=bands,
                    limit=limit,
                    percent_pixels=percent_pixels,
                    device=device,
                    dtype=torch.float32
    )
    ds.prepare_all_cases()

    model = CfcModelV7(input_size=8,
                            hidden_size=hidden_size, 
                            timeless_input_size=3,
                            sequence_length=sequence_length, 
                            output_size=len(bands), 
                            backbone_layers=backbone_layers, 
                            backbone_dropout=0,
                            activation=activation).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)

    train_subset, valid_subset = ds.get_train_validation_subset(0.2)
    print(f"Train dataset length: {len(train_subset)}")
    print(f"Validation dataset length: {len(valid_subset)}")
    #train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_loader = MemoryDataLoader(ds, batch_size=batch_size, indexes=torch.tensor(train_subset.indices).to(device=device, dtype=torch.int32))

    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    time0=time.time()
    for i,batch in enumerate(train_loader):
        if ((i+1)%100)==0:
            print(f"GPU {gpu_i}, Batch {i}")
            break
        (y, x, timeless, timespans) = batch
        optimizer.zero_grad()
        outputs = model(x, timeless, timespans)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    time1=time.time()
    print(f"Profiler example, GPU: {gpu_i}, Time for 100 batches: {time1-time0:.2f} s, {(100)/(time1-time0):.1f} batches/s")

    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    #prof.export_chrome_trace("trace.json")

if __name__ == "__main__":
    #playground()
    profiler_example()