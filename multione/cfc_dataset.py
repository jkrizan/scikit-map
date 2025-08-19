#%%

from calendar import c
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any
from numba import njit, prange
import numpy as np
from numpy.typing import NDArray
import pympler.asizeof
import torch
from torch.utils.data import Dataset,DataLoader
from pathlib import Path    
import zarr
from datetime import datetime, timedelta
import time
import pympler
from tqdm import tqdm

from settings import n_imag_per_year,n_threads
from utils import ttprint

fn_zarr = Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.zarr")
fn_zarr = Path(f"/data/oemc/arcov2/sample_v1.zarr")
#years = np.arange(2000, 2024)

#%%
'''
class ArcoV2Dataset(Dataset):
    pass
self = ArcoV2Dataset()
sequence_length = 12
'''


@njit(parallel=True, fastmath=True, cache=True, nogil=True)    
def _process_one_pixel(y, x, timespans, j_dates, j_lsdata, j_msdata, j_gtemp, sequence_length):
        nts = y.shape[0]
        nlsdata = j_lsdata.shape[0]
        for i in prange(nts):
            # i=0
            timespans[i, :] = (j_dates[i+1: i+1+sequence_length] - j_dates[i:i+sequence_length])/366
            y[i,:] = j_lsdata[:, i+sequence_length] # last value in sequence
            x[i, :, 0:nlsdata] = j_lsdata[:, i:i+sequence_length].T
            x[i, :, nlsdata] = j_msdata[i:i+sequence_length]
            x[i, :, nlsdata+1] = j_gtemp[i:i+sequence_length]
        
            # shape = 12,9

# @njit(parallel=True, fastmath=True, cache=True, nogil=True)
# def _calc_valid_values(lsdata, all_valid_values):
#     for i in prange(lsdata.shape[1]):
#         for j in range(lsdata.shape[2]):
#             # i=0; j=0
#             # all_valid_values[i, j] = np.isfinite(lsdata[:, i, j]).all() and np.isfinite(msdata[:, i, j]).all() and np.isfinite(gtemp[:, i, j]).all()
#             all_valid_values[i, j] = np.isfinite(lsdata[:, i, j]).all() 

class ArcoV2Dataset(Dataset):
                   
    def _read_tile_from_zarr(self, zarr_path, tile):
        # tile = self.tiles[0]
        dataset: zarr.Group = zarr.open(zarr_path, mode='r') #type: ignore
        group: zarr.Group = dataset[tile]   #type: ignore

        # list(group.array_keys())
        #   ['geom_temp_doy', 'covariates', 'modis', 'lsdata']
        # list(group.attrs.keys()) 
        #   ['tile', 'n_valid_pixels', 'n_sampled_pixels', 'time']

        #ttprint(f'Reading tile {tile}')
        #start = time.time()
        covariates: NDArray = group['covariates'][:] #type: ignore
        lsdata: NDArray = group['lsdata'][:]    #type: ignore
        msdata: NDArray = group['modis'][:]      #type: ignore
        gtemp: NDArray = group['geom_temp_doy'][:]  #type: ignore

        covariates[np.isnan(covariates)] = 0
        gtemp[np.isnan(gtemp)] = 0

        del dataset, group

        npixels = lsdata.shape[2]
        ##ttprint(f'Tile {tile} reading done in {time.time() - start:.2f} seconds')
        #all_valid_values = np.isnan(lsdata).sum(axis=0)==0 #np.isfinite(msdata[:,i]) & np.isfinite(lsdata[0,:,i])  
        
        all_valid_values = np.isfinite(lsdata).all(axis=0)# & np.isfinite(msdata).all(axis=0) & np.isfinite(gtemp).all(axis=0)

        #all_valid_values = np.empty((lsdata.shape[1], lsdata.shape[2]), dtype=bool)
        #_calc_valid_values(lsdata, all_valid_values)

        #start=time.time()
        data=[]
        meta=[]
        for j in range(npixels):
            # j=0
            valid_values = all_valid_values[:, j]   #type: ignore
            #if valid_values.sum() < sequence_length*2:
            #    continue                
            nvv = valid_values.sum()
            nts = nvv - self.sequence_length

            j_dates = self.days_from_start[valid_values] #self.dates[valid_values]
            j_lsdata = lsdata[:, valid_values, j]
            j_msdata = msdata[valid_values, j]
            j_gtemp = gtemp[self.ind_doys[valid_values], j]

            y = np.empty((nts, self.n_output_bands), dtype=np.float32)
            timespans = np.empty((nts, self.sequence_length), dtype=np.float32) # start, end
            x = np.empty((nts, self.sequence_length, self.n_features), dtype=np.float32) # bands + geom temp min/max
            x_timeless = covariates[:,j] # covariates for this pixel, shape = 17

            # for i in range(nts):
            #     # i=0
            #     timespans[i, :] = (j_dates[i+1: i+1+self.sequence_length] - j_dates[i:i+self.sequence_length])/366
            #     y[i] = j_lsdata[:, i+self.sequence_length] # last value in sequence
            #     x[i] = np.concatenate([
            #         j_lsdata[:, i:i+self.sequence_length].T, 
            #         j_msdata[i:i+self.sequence_length].reshape(-1,1),
            #         j_gtemp[i:i+self.sequence_length].reshape(-1,1),
            #     ], axis=-1)
                # shape = 12,9

            _process_one_pixel(y, x, timespans, j_dates, j_lsdata, j_msdata, j_gtemp, self.sequence_length)
            x[np.isnan(x)] = 0

            data.append((torch.tensor(y), torch.tensor(x), torch.tensor(x_timeless), torch.tensor(timespans)))
            meta.append((j,tile))

        #ttprint(f'Tile {tile} processing done in {time.time() - start:.2f} seconds, {len(data)} pixels')
        return data, meta
    '''
    class ArcoV2Dataset():
        pass
    self = ArcoV2Dataset()
    '''

    def __init__(self, zarr_path, years, sequence_length: int, limit=None):
        # years = np.arange(2020,2024); sequence_length = 12; limit=None; zarr_path = Path(f"/data/oemc/arcov2/sample_v1.zarr")
        if zarr_path is None:
            zarr_path = fn_zarr

        self.zarr_path = zarr_path
        self.years = years
        self.sequence_length = sequence_length
        self.n_output_bands = 7
        self.n_features = self.n_output_bands + 2 # modis_ndvi, geom temp
        
        dates_list=[]
        for y in years:
            dates = [datetime(y,1,8)+timedelta(days=16*i) for i in range(n_imag_per_year)]
            dates_list.extend(dates)        
        self.days_from_start = np.array([(d - datetime(years[0],1,1)).days + 1 for d in dates_list])
        #doys = list(range(8, 366, 16))*len(years)          
        self.dates = np.array(dates_list).astype('datetime64[D]')
        self.ind_doys = np.arange(len(self.dates)) % n_imag_per_year
        del dates_list

        self.dataset: zarr.Group = zarr.open(zarr_path, mode='r') #type: ignore
        self.tiles = list(self.dataset.group_keys())
        if limit is not None:
            self.tiles = self.tiles[:limit]

        self.data = []
        self.meta = []
        with ThreadPoolExecutor(max_workers= n_threads) as executor:
            futures=[executor.submit(self._read_tile_from_zarr,self.zarr_path,tile) for tile in self.tiles]
            for future in tqdm(as_completed(futures), total=len(futures), desc='Reading tiles'):
                data, meta = future.result()    #type: ignore
                self.data.extend(data)
                self.meta.extend(meta)

        self.length = len(self.data)
        self.n_features = self.data[0][1].shape[-1]  # number of features in x
        self.n_output_bands = self.data[0][0].shape[-1]  # number of output bands
        self.n_timeless_features = self.data[0][2].shape[-1]  # number of timeless features

        # pročitati sve pixele iz tileova (mislim da mi nije bitno koji je iz kojeg)
        # pripremiti za svaki pixel: 
        #       sequence bandova dimenzije input = [broj timeserija, sequence_length, broj bandova + geom temp min/max = 7 + 2]
        #       iz 'covariates' pokupiti sve vrijednosti za taj pixel dim=17 -1 dimenzionalan
        #       timespans = izvući van iz samih serija i years i .....
        # Napraviti random indeks svih pixela
        # Vraćati ih tim redom
        # Pogledati kako da taj random index nakon svake epohe resetiramo ....
        
    # def split_test_dataset(self, test_size: float = 0.2) -> ArcoV2Dataset:
    #     """
    #     Splits the dataset into a test dataset.
    #     :param test_size: Fraction of the dataset to be used as test set.
    #     :return: A new ArcoV2Dataset instance containing the test data.
    #     """

    #     n_test = int(self.length * test_size)
    #     all_inds = range(self.length)
    #     mask = np.zeros(self.length, dtype=bool)
    #     test_inds = np.random.choice(all_inds, size=n_test, replace=False)

    #     test_data = [self.data[i] for i in test_inds]
    #     test_meta = [self.meta[i] for i in test_inds]
    #     return ArcoV2Dataset(self.zarr_path, self.sequence_length, data=test_data, meta=test_meta)
    

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        return self.data[idx], self.meta[idx]
    
 


#%%
class ArcoV2DataLoader:
    def __init__(self, dataset: ArcoV2Dataset, inds: NDArray, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.inds = inds
        self.shuffle = shuffle
        self.length = len(inds)

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Any:
        if self.shuffle:
            np.random.shuffle(self.inds)
        self._current_index = 0
        return self

    def __next__(self) -> tuple[int, Any]:
        if self._current_index < self.length:
            idx = self.inds[self._current_index]
            self._current_index += 1
            data, _ = self.dataset[idx]
            y, x, timeless_x, timespans = data
            # x[x.isnan()] = 0
            # timeless_x[timeless_x.isnan()] = 0
            return self._current_index, (y, x, timeless_x, timespans)
        else:
            raise StopIteration

class ArcoV2DataLoaderFactory:
    def __init__(self, dataset: ArcoV2Dataset, validation_size: float, random_seed:int):
        self.dataset = dataset
        self.validation_size = validation_size
        self.rs = np.random.RandomState(random_seed)

        all_inds = self.rs.permutation(np.arange(len(dataset)))

        n_val = int(len(dataset) * validation_size)

        self.train_inds = all_inds[:-n_val]
        self.val_inds = all_inds[-n_val:]

    def get_train_loader(self) -> ArcoV2DataLoader:
        return ArcoV2DataLoader(self.dataset, self.train_inds)

    def get_val_loader(self) -> ArcoV2DataLoader:
        return ArcoV2DataLoader(self.dataset, self.val_inds)

    
#%%
def testing():
    #%%
    ds = ArcoV2Dataset(fn_zarr, sequence_length=12)

    print(f'{len(ds)} - {pympler.asizeof.asizeof(ds)/1024**2:<.2f} MB')
    nts = 0
    for i in tqdm(range(len(ds))):
        batch, meta = ds[i]
        nts = nts + batch[0].shape[0]
        #print(f"batch {i} - {meta[0]} - {meta[1]}, {batch[1].shape}")

    print(f'Total number of timeseries: {nts}')

if __name__=='__main__':
    testing()