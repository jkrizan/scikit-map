#%%

from concurrent.futures import ThreadPoolExecutor, as_completed
from os import read
import re
from numpy.ma import indices
import torch
from torch import Tensor, nn
from typing import List, Optional, Self, Union, Any
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
import threading
from queue import Queue

class MemoryDataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, indexes: ArrayLike | None, cache_length: int = 0, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        if indexes is None:
            indexes = np.array(range(len(self.dataset))) # type: ignore
        self.indexes = indexes
        self.max_idx = (len(self.indexes)-1)//self.batch_size + 1   # type: ignore
        self.batch_idx = 0
        self.queue: Optional[Queue] = None
        self.cache_length = cache_length
        self.shuffle = shuffle
        self.setup_queue()
        self.rng = np.random.default_rng()

    def __iter__(self):
        return self
    
    def __len__(self):
        return self.max_idx

    def __next__(self):
        if self.queue is not None:                        
            batch =  self.queue.get(
                block=True, timeout=None
            )   # type: ignore
        else:
            batch = self.get_batch()

        if batch is None:
            self._setup_iteration()
            raise StopIteration
        
        return batch

    def _setup_iteration(self):
        self.batch_idx = 0
        self.setup_queue()
        if self.shuffle:
            self.shuffler()

    def setup_queue(self):
        if self.cache_length>0:
            #self.iteration_ended = False
            self.queue = Queue(maxsize=self.cache_length)
            threading.Thread(target=self._queue_thread, daemon=True).start()
        else:
            self.queue = None

    def shuffler(self):   
        indices = self.indexes.cpu().numpy() if isinstance(self.indexes, torch.Tensor) else self.indexes     
        self.rng.shuffle(indices)
        self.indexes = torch.tensor(indices, device=self.indexes.device) if isinstance(self.indexes, torch.Tensor) else indices


    def _queue_thread(self):
        while True:
            batch = self.get_batch()
            self.queue.put(batch)     # type: ignore
            if batch is None:
                self.iteration_ended = True
                break        
        
    def get_batch(self):
        if self.batch_idx >= self.max_idx:
            return None
        else:
            start = self.batch_idx * self.batch_size
            end = start + self.batch_size
            batch_indexes = self.indexes[start:end] # type: ignore
            self.batch_idx += 1

            return self.dataset.get_cases(batch_indexes)    # type: ignore
        
        

class ArcoV2Dataset(Dataset):

    def clone_to(self, device: str) -> Self:
        if self.prepared_all_cases:
            new_self = copy.copy(self)
            new_self.device = device
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
                 indices: list[str],
                 limit=None, 
                 percent_pixels: float = 1.0, 
                 device: str = 'cpu', 
                 dtype=torch.float32,
                 return_tensors: bool = True):
        
        self.prepared_all_cases = False
        self.percent_pixels = percent_pixels
        self.nthreads = 4
        self.zarr_path = zarr_path
        self.years = years
        self.sequence_length = sequence_length
        self.indices = indices
        self.n_output = len(self.indices)
        self.n_features = self.n_output + 2 # modis_ndvi, geom temp
        self.return_tensors = return_tensors

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
            dataset: zarr.Group = zarr.open(self.zarr_path, mode='r') #type: ignore
            tiles = list(dataset.group_keys())
            tiles = sorted(tiles)
            generator = np.random.default_rng(46)
            self.tiles = generator.permutation(tiles) # to get always same order of tiles
            if limit is not None:
                if isinstance(limit, int):
                    self.tiles = self.tiles[:limit]
                elif isinstance(limit, (list, tuple)):
                    self.tiles = self.tiles[limit[0]:limit[1]]
                elif (isinstance(limit, slice)):
                    self.tiles = self.tiles[limit]
    
            self.data: List = [None] * len(self.tiles)
            self.pixel_indices_list: List = []
            self.ncases = 0
            self.npixels = 0
            #scaler = StandardScaler() 
            with ThreadPoolExecutor(max_workers= self.nthreads) as executor:
                futures=[executor.submit(self._read_tile_from_zarr,tile, tj) for tj, tile in enumerate(self.tiles)]
                for future in tqdm(as_completed(futures), total=len(futures), desc='Reading tiles'):
                    tj, tile, tile_data = future.result()    #type: ignore                    
                    # tj, tile, tile_data = self._read_tile_from_zarr(self.zarr_path, self.tiles[1], 1)
                    
                    tile_nts = tile_data[0]
                    if len(tile_nts)==0:
                        data: List = [None for d in tile_data[1:]] + [tile_nts]
                    else:
                        for pj in range(tile_nts.shape[0]):
                            self.pixel_indices_list.extend([(tj, pj, jts) for jts in range(tile_nts[pj] - 1)])
                            self.ncases += tile_nts[pj] - 1
                            
                        data: List = [torch.tensor(d, dtype=self.dtype) for d in tile_data[1:-1]]
                        if tile_data[-1] is not None:
                            data.append(tile_data[-1].astype(bool)) # all_valid_values as boolean
                        else:
                            data.append(None)
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

    def _read_tile_from_zarr(self, tile, tj:int):
        # tile = self.tiles[0]; tj=0
        dataset: zarr.Group = zarr.open(self.zarr_path, mode='r') #type: ignore
        group: zarr.Group = dataset[tile]   #type: ignore
        group_arrays = list(group.array_keys())

        pixel_inds = group['pixel_inds'][:]   #type: ignore
        if 'water_mask' not in group_arrays:
            print(f"Tile {tj}. {tile} has no water mask, skipping")
            return tj, tile, (np.array([]), None, None, None, None, None, None)
        
        water_mask: NDArray = group['water_mask'][:].flatten()[pixel_inds]   #type: ignore
        valid_inds = water_mask < 95    # 127 is missing value
        if not np.any(valid_inds):
            return tj, tile, (np.array([]), None, None, None, None, None)

        lsdata: NDArray = group['lsdata'][:,:,valid_inds]    #type: ignore
        msdata: NDArray = group['modis'][:,valid_inds]      #type: ignore
        gtemp: NDArray = group['geom_temp_doy'][:,valid_inds]  #type: ignore
        valid_values_mask: NDArray = group['valid_values_mask'][:, valid_inds]  #type: ignore
        gtemp_min = gtemp.min(axis=0)
        gtemp_max = gtemp.max(axis=0)

        del dataset, group

        valid_values_mask[np.isnan(msdata)]=False

        npixels = lsdata.shape[2]   # number of sampled pixels in one tile
        if self.percent_pixels<1.0:
            npixels = int(npixels*self.percent_pixels)
            generator = np.random.default_rng(46)
            inds = generator.permutation(npixels)[:npixels]
            lsdata = lsdata[:,:,inds]
            msdata = msdata[:,inds]
            gtemp = gtemp[:,inds]
            valid_values_mask = valid_values_mask[:,inds]            

        indices_data = np.empty((self.n_output, lsdata.shape[1], lsdata.shape[2]), dtype=np.float32)
        for i, ind in enumerate(self.indices):
            red = lsdata[0]; nir = lsdata[1]; blue = lsdata[2]; green = lsdata[3]; swir1 = lsdata[4]; swir2 = lsdata[5]
            if ind == 'ndvi':
                denom = (nir + red)
                denom[denom == 0] = np.nan
                indices_data[i] = (nir - red) / denom
                valid_values_mask[np.isnan(indices_data[i])] = False
            elif ind == 'fpar':
                # f'(((({ndvi_form} - ndvi_min)*(fpar_max - fpar_min))/(ndvi_max - ndvi_min)) + fpar_min)'
                denom = (nir + red)
                denom[denom == 0] = np.nan
                ndvi = (nir - red) / denom
                ndvi_min = 0.03
                ndvi_max = 0.96
                fpar_min = 0.001
                fpar_max = 0.95
                scale = (fpar_max - fpar_min) / (ndvi_max - ndvi_min)
                indices_data[i] = np.clip((ndvi - ndvi_min) * scale + fpar_min, 0, 1)
                valid_values_mask[np.isnan(indices_data[i])] = False
            # TODO: fix other indices
            # elif ind == 'evi':
            #     indices_data[i] = 2.5 * (lsdata[4] - lsdata[3]) / (lsdata[4] + 6 * lsdata[3] - 7.5 * lsdata[1] + 1)
            # elif ind == 'savi':
            #     indices_data[i] = 1.5 * (lsdata[4] - lsdata[3]) / (lsdata[4] + lsdata[3] + 0.5)
            # elif ind == 'msavi2':
            #     indices_data[i] = (2 * lsdata[4] + 1 - np.sqrt((2 * lsdata[4] + 1)**2 - 8 * (lsdata[4] - lsdata[3]))) / 2
            # elif ind == 'nbr':
            #     indices_data[i] = (lsdata[4] - lsdata[5]) / (lsdata[4] + lsdata[5] + 1e-6)
            # elif ind == 'ndwi':
            #     indices_data[i] = (lsdata[2] - lsdata[4]) / (lsdata[2] + lsdata[4] + 1e-6)
            else:
                raise Exception(f"Index {ind} not recognized")

        nts = valid_values_mask.sum(axis=0) - self.sequence_length
        invalid_pixels = nts < 10
        if invalid_pixels.any():
            lsdata = lsdata[:,:,~invalid_pixels]
            msdata = msdata[:,~invalid_pixels]
            gtemp = gtemp[:,~invalid_pixels]
            gtemp_min = gtemp_min[~invalid_pixels]
            gtemp_max = gtemp_max[~invalid_pixels]
            valid_values_mask = valid_values_mask[:,~invalid_pixels]
            indices_data = indices_data[:,:,~invalid_pixels]
            nts = nts[~invalid_pixels]                        
       
        data=[nts, indices_data, msdata, gtemp, gtemp_min, gtemp_max, valid_values_mask]

        return tj, tile, tuple(data)

    def get_cases(self, indices: List|np.ndarray) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        y = self.all_y[indices]
        x = self.all_x[indices]
        x_gtemp = self.all_x_gtemp[indices]
        timespans = self.all_timespans[indices]
        return (y, x, x_gtemp, timespans)

    def get_one_case(self, idx: int):
        
        if self.prepared_all_cases:

            if self.return_tensors:
                return (self.all_y[idx], self.all_x[idx], self.all_x_gtemp[idx], self.all_timespans[idx])
            else:
                return (self.all_y[idx].cpu().numpy(), self.all_x[idx].cpu().numpy(), self.all_x_gtemp[idx].cpu().numpy(), self.all_timespans[idx].cpu().numpy())

        else:
            raise Exception("Data not prepared, cannot get one case")
    
    def prepare_all_cases(self):
        
        @njit(parallel=True)
        def process_tile(y, x, x_gtemp, timespans_data, pixel_indices,
                         indata, msdata,
                         gtemp, gtemp_min, gtemp_max,
                         all_valid_values, tile_nts,
                         ind_doys, days_from_start,
                         sequence_length) -> None:
            npix = tile_nts.shape[0]
            ncases = (tile_nts - 1).cumsum()
            nbands = indata.shape[0]
            #ncase = 0
            for pix_ind in prange(npix):
                # pix_ind=0
                valid_values = all_valid_values[:, pix_ind]                
                j_dates = days_from_start[valid_values]
                j_lsdata = indata[:,valid_values, pix_ind]
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
            (indata, msdata, gtemp, gtemp_min, gtemp_max, all_valid_values, tile_nts) = self.data[t]
            if len(tile_nts)==0:
                continue
            ncases = (tile_nts - 1).sum()
            y = np.empty((ncases, self.n_output, 2), dtype=dtype)
            x = np.empty((ncases, self.sequence_length, self.n_features), dtype=dtype)
            x_gtemp = np.empty((ncases, 3), dtype=dtype)
            timespans_data = np.empty((ncases, self.sequence_length + 1), dtype=dtype)
            pixel_indices = np.empty((ncases,), dtype=np.int32)

            indata = indata.to(ttype).numpy()
            msdata = msdata.to(ttype).numpy()
            gtemp = gtemp.to(ttype).numpy()
            #all_valid_values = all_valid_values.to(torch.bool).numpy()
            days_from_start = self.days_from_start.to(torch.int16).numpy()
            sequence_length = self.sequence_length
            ind_doys = self.ind_doys.to(torch.int16).numpy()
            gtemp_min = gtemp_min.to(ttype).numpy()
            gtemp_max = gtemp_max.to(ttype).numpy()

            process_tile(y, x, x_gtemp, timespans_data, pixel_indices,
                     indata, msdata,
                     gtemp, gtemp_min, gtemp_max,
                     all_valid_values, tile_nts,
                     ind_doys, days_from_start,
                     sequence_length)

            y = torch.tensor(y, dtype=self.dtype).to(self.device)
            x = torch.tensor(x, dtype=self.dtype).to(self.device)
            x_gtemp = torch.tensor(x_gtemp, dtype=self.dtype).to(self.device)            
            timespans_data = torch.tensor(timespans_data, dtype=self.dtype).to(self.device)

            tiles.append((y, x, x_gtemp, timespans_data))

            del indata, msdata, gtemp, all_valid_values, tile_nts
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

    def get_train_validation_subset(self, ncases_validation: float):
        indices = np.array(range(self.length))
        rndgen = np.random.default_rng(46)
        rndgen.shuffle(indices)
        ncases_validation = int(ncases_validation * len(indices))
        train_subset = Subset(self, indices[ncases_validation:].tolist())
        valid_subset = Subset(self, indices[:ncases_validation].tolist())
        return train_subset, valid_subset

    def get_train_validation_indices(self, ncases_validation: float):
        indices = np.array(range(self.length))
        rndgen = np.random.default_rng(46)
        rndgen.shuffle(indices)
        ncases_validation = int(ncases_validation * len(indices))
        return indices[ncases_validation:], indices[:ncases_validation]   

    def prepare_for_ray_worker(self, rank: int, nranks: int, ncases_validation: float, device: str|torch.device) -> tuple[Self, np.ndarray, np.ndarray]:
        train_indices, val_indices = self.get_train_validation_indices(ncases_validation=ncases_validation)
        train_indices = train_indices[rank::nranks]
        val_indices = val_indices[rank::nranks]
        all_indices = np.concatenate((val_indices, train_indices))

        new_self = copy.copy(self)
        new_self.device = device
    
        new_self.all_y = self.all_y[all_indices,...].to(device)
        new_self.all_x = self.all_x[all_indices,...].to(device)
        new_self.all_x_gtemp = self.all_x_gtemp[all_indices,...].to(device)
        new_self.all_timespans = self.all_timespans[all_indices,...].to(device)

        return new_self, np.arange(len(val_indices)), np.arange(len(train_indices))

    def get_nontraining_subset(self, ncases_validation: float, ncases: int):
        # get subset for testing, not used in training or validation
        indices = np.array(range(self.length))
        rndgen = np.random.default_rng(46)
        rndgen.shuffle(indices)
        ncases_validation = int(ncases_validation * len(indices))
        ncases = min(ncases, ncases_validation)
        subset = Subset(self, indices[:ncases_validation][:ncases].tolist())
        return subset

    def set_subset(self, subset: NDArray):
        self.subset = subset
        self.length = len(subset)

    def __len__(self) -> int:
        if self.prepared_all_cases:
            return self.all_y.shape[0]
        else:
            return self.length

    def __getitem__(self, idx: int):
        idx = self.subset[idx]
        return self.get_one_case(idx)

class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)

class CfcModel(nn.Module):
    def __init__(self, 
                 input_size:int, 
                 timeless_input_size: int,
                 hidden_size: int, 
                 activation: str = 'lecun_tanh', #silu, relu, tanh, gelu, lecun_tanh
                 sequence_length:int = 12, 
                 backbone_layers:list[int]=[64, 64, 64],
                 backbone_dropout: float = 0.0, 
                 output_size:int = 1,
                 ):

        super(CfcModel, self).__init__()

        self.input_size = input_size
        self.timeless_input_size = timeless_input_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.backbone_layers = backbone_layers
        self.backbone_dropout = backbone_dropout
        self.output_size = output_size
        self.activation=activation

        self.mode='default'

        self.rnn = CfCCell(
                self.input_size+self.timeless_input_size,
                self.hidden_size,
                self.mode,
                self.activation,
                self.backbone_layers,
                self.backbone_dropout,
                )
        self.lstm = LSTMCell(self.input_size+self.timeless_input_size, self.hidden_size)  # Mixed memory        

        fc_size = self.hidden_size #+ 2 * self.output_size #
        self.fc_fwd = nn.Sequential(
            nn.Linear(fc_size, fc_size//2),
            LeCun(),
            nn.Linear(fc_size//2, fc_size//4),
            LeCun(),
            nn.Linear(fc_size//4, fc_size//8),
            LeCun(),
            nn.Linear(fc_size//8, self.output_size)
        )        

        self.fc_bwd = nn.Sequential(
            nn.Linear(fc_size, fc_size//2),
            LeCun(),
            nn.Linear(fc_size//2, fc_size//4),
            LeCun(),
            nn.Linear(fc_size//4, fc_size//8),
            LeCun(),
            nn.Linear(fc_size//8, self.output_size)
        )    
            
        self.init_weights()

    
    def inference(self, x, timeless, timespans, direction:str):
        #device = x.device
        dtype = x.dtype
        batch_size, seq_len = x.shape[:2]
        #x_mean = x[:,:,:self.output_size].mean(dim=1).detach()
        #x_std = x[:,:,:self.output_size].std(dim=1).detach()

        if direction=='forward' or direction=='both':           
            h_state = torch.zeros((batch_size, self.hidden_size), dtype=dtype)
            c_state = torch.zeros((batch_size, self.hidden_size), dtype=dtype)

            for t in range(seq_len):            
                inputs = torch.cat((x[:, t, :].squeeze(1), timeless), dim=1)
                
                ts = timespans[:, t + 1].reshape(-1,1)

                h_state, c_state = self.lstm(inputs, (h_state, c_state))
                h_out_fw, h_state = self.rnn(inputs, ts, hx=h_state)

            #merged_fwd = torch.cat([h_out_fw, x_mean, x_std], dim=1)    #type: ignore

        if direction=='backward' or direction=='both':
            h_state = torch.zeros((batch_size, self.hidden_size), dtype=dtype)
            c_state = torch.zeros((batch_size, self.hidden_size), dtype=dtype)

            for t in reversed(range(seq_len)):
                inputs = torch.cat((x[:, t, :].squeeze(1), timeless), dim=1)            

                ts = -1 * timespans[:, t].reshape(-1,1)

                h_state, c_state = self.lstm(inputs, (h_state, c_state))
                h_out_bw, h_state = self.rnn(inputs, ts, hx=h_state)            

        if direction=='forward':
            return self.fc_fwd(h_out_fw) # type: ignore
        elif direction=='backward':
            return self.fc_bwd(h_out_bw) # type: ignore
        elif direction=='both':
            readout = np.concatenate((self.fc_fwd(h_out_fw).unsqueeze(2), self.fc_bwd(h_out_bw).unsqueeze(2)), axis=2) #type: ignore
            return readout

    def forward(self, x, timeless, timespans):
        # x (batch_size, 1, seq_len, input_size)
        device = x.device
        dtype = x.dtype

        batch_size, seq_len = x.size(0), x.size(1)            

        # forward pass
        h_state = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
        c_state = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)

        # forward pass
        for t in range(seq_len):            
            inputs = torch.cat((x[:, t, :].squeeze(1), timeless), dim=1)
            
            ts = timespans[:, t + 1].reshape(-1,1)

            h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out_fw, h_state = self.rnn(inputs, ts, hx=h_state)


        # backward pass               
        h_state = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)
        c_state = torch.zeros((batch_size, self.hidden_size), device=device, dtype=dtype)

        for t in reversed(range(seq_len)):
            inputs = torch.cat((x[:, t, :].squeeze(1), timeless), dim=1)            

            ts = -1 * timespans[:, t].reshape(-1,1)

            h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out_bw, h_state = self.rnn(inputs, ts, hx=h_state)

        
        res_fwd = self.fc_fwd(h_out_fw) # type: ignore
        res_bwd = self.fc_bwd(h_out_bw) # type: ignore
        readout = torch.cat((res_fwd.unsqueeze(2), res_bwd.unsqueeze(2)), dim=2) #type: ignore

        return readout


    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w, generator=torch.Generator())
            else:
                torch.nn.init.uniform_(w, generator=torch.Generator())

    

# %%
