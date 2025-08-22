#%%

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from re import A
from typing import Any
from zipfile import ZipFile
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
import copy

from settings import n_imag_per_year,n_threads
from utils import ttprint

#fn_zarr = Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.zarr")
#fn_zarr = Path(f"/data/oemc/arcov2/sample_v1.zarr")
#years = np.arange(2000, 2024)

#%%
'''
class ArcoV2Dataset(Dataset):
    pass
self = ArcoV2Dataset()
sequence_length = 12
'''

class ArcoV2DatasetV2(Dataset):
    def __init__(self, zarr_path, years, sequence_length: int, limit=None, read_timeless=False, device=torch.get_default_device()):
        # years = np.arange(2020,2024); sequence_length = 12; limit=None; zarr_path = Path(f"/data/oemc/arcov2/sample_v1.zarr")
        #if zarr_path is None:
        #    zarr_path = fn_zarr

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
            self.tiles = list(dataset.group_keys())
            if limit is not None:
                self.tiles = self.tiles[:limit]

            self.timeless_data=[None] * len(self.tiles)
            self.data = [None] * len(self.tiles)
            self.pixel_indices = []
            self.ncases = 0
            self.npixels = 0
            with ThreadPoolExecutor(max_workers= n_threads) as executor:
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
                self.n_timeless_features = self.data[0][5].shape[-1]            
            
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


        if self.read_timeless:
            data=(nts, lsdata, msdata, gtemp, all_valid_values, covariates) # type: ignore
        else:
            data=(nts, lsdata, msdata, gtemp, all_valid_values)

        return tj, tile, data

    def get_one_case(self, idx: int):
        tile_ind, pix_ind, ts_ind = self.pixel_indices[idx]
        (nts, lsdata, msdata, gtemp, all_valid_values, *rest) = self.data[tile_ind]
        if self.read_timeless:
            covariates = rest[0]

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

        with torch.device(self.device):
            res = [torch.tensor(y), torch.tensor(x), torch.tensor(timespans)]
            if self.read_timeless:
                x_timeless = covariates[:,pix_ind]  # type: ignore
                res.append(torch.tensor(x_timeless))

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
        return self.get_one_case(idx)
#%%

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
        # tile = self.tiles[7]
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
        gtemp = group['geom_temp_doy'][:]  #type: ignore

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
        #if zarr_path is None:
        #    zarr_path = fn_zarr

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

        if self.zarr_path is not None:
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
    
    @classmethod
    def from_arco(cls, arco_path: Path, years: NDArray, sequence_length:int, n_threads:int=16, limit=None, device=None):
        # arco_path = Path('/home/josip/arcov2/sample_v1.arco')
        # Implement logic to read ARCO format files and create dataset
        
        import tqdm
        from zipfile import ZipFile
        from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

        # count number of pixels
        # npixels = 0
        data=[]; meta=[]; tiles=[]
        files = list(arco_path.glob('*.arco'))     
        if limit is not None:
            files=files[-limit:]   

        def _read_tile_from_zip(tile_path: Path):
            tile = tile_path.stem
            ldata=[]; lmeta=[]
            with ZipFile(tile_path, 'r') as zipf:
                for file in zipf.namelist():
                    if file.endswith('.npz'):
                        with zipf.open(file) as f:
                            npz_data = np.load(f)
                            y = npz_data['y']
                            x = npz_data['x']
                            timeless_x = npz_data['timeless_x']
                            timespans = npz_data['timespans']
                            if device is not None:
                                ldata.append((torch.tensor(y, device=device), torch.tensor(x, device=device), torch.tensor(timeless_x, device=device), torch.tensor(timespans, device=device)))
                            else:
                                ldata.append((torch.tensor(y), torch.tensor(x), torch.tensor(timeless_x), torch.tensor(timespans)))
                            lmeta.append((int(file.split('.')[0]), tile))
            return ldata, lmeta

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []
            for tile_path in files:
                # tile_path = files[0]
                tile = tile_path.stem
                tiles.append(tile)
                futures.append(executor.submit(_read_tile_from_zip, tile_path))
            
            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc='Reading tiles'):
                tile_data, tile_meta = future.result()
                data.extend(tile_data)
                meta.extend(tile_meta)

            
        ds = cls(None, years, sequence_length=sequence_length)
        ds.data = data
        ds.meta = meta
        ds.length = len(data)
        ds.n_features = ds.data[0][1].shape[-1]  # number of features in x
        ds.n_output_bands = ds.data[0][0].shape[-1]  # number of output bands
        ds.n_timeless_features = ds.data[0][2].shape[-1]
        return ds

    @classmethod
    def from_one_tile(cls, tile, years, data, valid_data, sequence_length=12, num_of_pixels=0.2) -> "ArcoV2Dataset":
        (n_valid_pixels, inds_valid_pixels, valid_values_mask) = valid_data
        (landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy) = data

        
        ds = cls(None, years, sequence_length=sequence_length)
        ds.data = []
        ds.meta = []
        covariate_data[np.isnan(covariate_data)] = 0
        geom_temp_doy[np.isnan(geom_temp_doy)] = 0
        npixels = landsat_data.shape[1]
        n_bands = ds.n_output_bands
        n_dates = len(ds.dates)
        ##ttprint(f'Tile {tile} reading done in {time.time() - start:.2f} seconds')
        #all_valid_values = np.isnan(lsdata).sum(axis=0)==0 #np.isfinite(msdata[:,i]) & np.isfinite(lsdata[0,:,i])  
        
        lsdata = np.empty((n_bands, n_dates, npixels), dtype=np.float32)  # type: ignore
        for b in range(n_bands):    # this is slow ...
            lsdata[b,:,:] = landsat_data[b*n_dates:(b+1)*n_dates, :]

        n_valid_values = np.isfinite(lsdata).sum(axis=0)
        all_valid_values = np.isfinite(lsdata).sum(axis=0)
        valid_pixels = np.where(n_valid_values > ds.sequence_length*4)[0]

        if num_of_pixels<=1:
            num_of_pixels = int(npixels * num_of_pixels)

        inds = np.random.choice(valid_pixels[:num_of_pixels], size=int(num_of_pixels), replace=False)

        for j in tqdm(inds):
            valid_values = all_valid_values[:, j]   #type: ignore                        
            nvv = valid_values.sum()
            nts = nvv - ds.sequence_length
            y = np.empty((nts, ds.n_output_bands), dtype=np.float32)
            timespans = np.empty((nts, ds.sequence_length), dtype=np.float32) # start, end
            x = np.empty((nts, ds.sequence_length, ds.n_features), dtype=np.float32) # bands + geom temp min/max
            x_timeless = covariate_data[:,j] #
            x[np.isnan(x)]= 0            

            ds.data.append((torch.tensor(y), torch.tensor(x), torch.tensor(x_timeless), torch.tensor(timespans) ))
            ds.meta.append((j, tile))

        return ds

#%%
class ArcoV2DataLoader:
    def __init__(self, dataset: ArcoV2Dataset, inds: NDArray, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.inds = inds
        self.shuffle = shuffle
        self.length = len(inds)
        self._current_index=0

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

def save_dataset_to_arco(dataset: ArcoV2Dataset,  output_path: Path) -> None:
    # output_path = Path('/home/josip/arcov2/sample_v1.arco')
    import tqdm
    from zipfile import ZipFile
    from concurrent.futures import ThreadPoolExecutor, as_completed

    n_threads = 16
    tiles, inds = np.unique(np.array([m[1] for m in dataset.meta]), return_inverse=True)

    def _write_to_zip(tile_path, inds):
        with ZipFile(tile_path, 'w') as zipf:
            for i in inds:
                data, meta = dataset.data[i], dataset.meta[i]
                y, x, timeless_x, timespans = data
                j, tile = meta
                np.savez_compressed(zipf.open(f'{j:07}.npz', 'w'), y=y.numpy(), x=x.numpy(), timeless_x=timeless_x.numpy(), timespans=timespans.numpy())
        return tile_path

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for i, tile in tqdm.tqdm(enumerate(tiles), desc='Saving dataset to ARCO format'):
            tile_path = (output_path / tile).with_suffix('.arco')
            tile_inds = np.where(inds == i)[0]

            futures.append(executor.submit(_write_to_zip, tile_path, tile_inds))

        for future in tqdm.tqdm(as_completed(futures),total=len(tiles)):
            tile = future.result()
            tqdm.tqdm.write(f'Saved tile: {tile}')

    #for data, meta in tqdm.tqdm(zip(dataset.data, dataset.meta), total=len(dataset.data), desc='Saving dataset to ARCO format'):
    # for i, tile in tqdm.tqdm(enumerate(tiles), desc='Saving dataset to ARCO format'):
    #     # i=0; tile = tiles[i]
        
    #     tile_path = (output_path / tile).with_suffix('.arco')
    #     zipf = 
    #     tile_inds = np.where(inds==i)[0]
    #     for i in tile_inds:
    #         data, meta = dataset.data[i], dataset.meta[i]
    #         y, x, timeless_x, timespans = data
    #         j, tile = meta
    #         np.savez_compressed(zipf.open(f'{j:07}.npz','w'), y=y, x=x, timeless_x=timeless_x, timespans=timespans)
    #     zipf.close()
        

def testing_1():
    #%%
    ds = ArcoV2Dataset(fn_zarr, sequence_length=12)

    print(f'{len(ds)} - {pympler.asizeof.asizeof(ds)/1024**2:<.2f} MB')
    nts = 0
    for i in tqdm(range(len(ds))):
        batch, meta = ds[i]
        nts = nts + batch[0].shape[0]
        #print(f"batch {i} - {meta[0]} - {meta[1]}, {batch[1].shape}")

    print(f'Total number of timeseries: {nts}')
#%%

def testing_2():
    fn_zarr = Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.zarr")
    ds = ArcoV2DatasetV2(fn_zarr, years=np.arange(2000, 2024), sequence_length=12, limit=10)


if __name__=='__main__':
    testing()