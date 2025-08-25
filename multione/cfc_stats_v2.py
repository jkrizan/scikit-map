#%%
from ctypes import util
import datetime
import numpy as np
import torch
from pathlib import Path
import datetime
from torch.utils.data import DataLoader

from cfc_v4 import CfcLearner_v4, ArcoV2DatasetV2
import cfc_sample
from cfc_dataset import _process_one_pixel
from multione.utils import n_pix
from settings import bands_prefix_out
from skmap import parallel
import utils
from utils import get_temperature_for_doy, get_temperature_for_year
from numba import njit, prange

#%%
#import importlib
# import utils
# utils = importlib.reload(utils)
#cfc_train = importlib.reload(cfc_train)
#%%
def test_timeseries():
#%%
    tile = '055W_06S'
    years = np.arange(2000, 2024)
    sequence_length = 12
    
    (success, error, eta), meta, valid_data, data = cfc_sample.get_tile_data(tile)    
    (n_valid_pixels, inds_valid_pixels, valid_values_mask) = valid_data
    (landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy) = data

    n_bands = landsat_data.shape[0] // (23 * len(years))
    n_output_bands = 7
    n_features = 9

    ds = ArcoV2DatasetV2(Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.zarr"),  years, sequence_length,limit=10, read_timeless=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

#%%
    fn_ckpt = Path('/mnt/nibble/gen_cog/arcov2/cfc-v4_e-84.ckpt')
    input_size = 9 #dataset.n_features
    output_size = 7 #dataset.n_output_bands
    sequence_length = 12
    #n_timeless_features = 17 #dataset.n_timeless_features
    model = CfcLearner_v4.load_from_checkpoint(fn_ckpt, map_location='cpu', strict=True) # input_size=input_size, sequence_length=12, output_size=output_size)
    model.freeze()

#%%
    pixel_ind = 10000000
    valid_values = valid_values_mask[:, pixel_ind]
    nvv = valid_values.sum()
    nts = nvv - sequence_length
    n_dates = ds.days_from_start.shape[0]
    j_dates = ds.days_from_start[valid_values]
    j_lsdata = np.empty((n_bands, nvv), dtype=np.float32)
    for b in range(n_bands):
        j_lsdata[b,:] = landsat_data[b*n_dates:(b+1)*n_dates, pixel_ind][valid_values]
    j_msdata = modis_data[valid_values, pixel_ind]    
    j_gtemp = geom_temp_doy[ds.ind_doys[valid_values], pixel_ind]

    y = np.empty((nts, n_output_bands), dtype=np.float32)
    timespans = np.empty((nts, sequence_length), dtype=np.float32)
    x = np.empty((nts, sequence_length, n_features), dtype=np.float32)  
    _process_one_pixel(y, x, timespans, j_dates, j_lsdata, j_msdata, j_gtemp, ds.sequence_length)
    timeless = covariate_data[:, pixel_ind]

    x[np.isnan(x)] = 0
    x[:,:,7]  = x[:,:,7]/10000
    x[:,:,8]  = x[:,:,8]/100
                                           

#%%
    # tile_data = dataset[1][0]
    # y, x, _, timespans = tile_data
    # y, x, timespans = y.detach().clone(), x.detach().clone(), timespans.detach().clone()
    # x[:,:,7]  = x[:,:,7]/10000
    # x[:,:,8]  = x[:,:,8]/100

    xx = torch.tensor(x)#.unsqueeze(1)
    tt = torch.tensor(timespans)#.unsqueeze(1)
    batchsize = xx.size(0)
    x_timeless = torch.tensor(timeless).expand(batchsize, -1)
    y_hat = model(xx, tt, x_timeless)

    print(torch.nn.MSELoss()(y_hat, torch.tensor(y)).item())
    y_hat = y_hat.detach().numpy()  

#%%
    dates = ds.dates[valid_values]
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(7, 1, figsize=(10, 30))
    for b in range(7):
        ax = axs[b]
        ax.plot(dates[sequence_length:], y[:,b],'bo', label='observed')
        ax.plot(dates[sequence_length:], y_hat[:,b], 'r.', label='predicted')
        ax.set_title(f"Band {bands_prefix_out[b]}")
        if b==0: 
            ax.legend()
    plt.show()
#%%
def test_whole_image():
    #%%
    tile = '055W_06S'
    years = np.arange(2000, 2024)
    sequence_length = 12
    
    (success, error, eta), meta, valid_data, data = cfc_sample.get_tile_data(tile)    
    (n_valid_pixels, inds_valid_pixels, valid_values_mask) = valid_data
    (landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy) = data

    n_bands = landsat_data.shape[0] // (23 * len(years))
    n_output_bands = 7
    n_features = 9

    ds = ArcoV2DatasetV2(Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.zarr"),  years, sequence_length,limit=10, read_timeless=True)
    #dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
#%%
    fn_ckpt = Path('/mnt/nibble/gen_cog/arcov2/cfc-v4_e-116.ckpt')
    input_size = 9 #dataset.n_features
    output_size = 7 #dataset.n_output_bands
    sequence_length = 12
    #n_timeless_features = 17 #dataset.n_timeless_features
    model = CfcLearner_v4.load_from_checkpoint(fn_ckpt, map_location='cpu', strict=True) # input_size=input_size, sequence_length=12, output_size=output_size)
    model.freeze()
#%%
    # predict whole image for some date
    sequence_length = ds.sequence_length
    year = 2020
    month = 6
    date = datetime.datetime(year, month, 15)
    doy = date.timetuple().tm_yday
    day_from_start = (date - datetime.datetime(years[0], 1, 1)).days - 1
    n_pixels = landsat_data.shape[1]
    n_dates = ds.days_from_start.shape[0]

    transform = meta[1]
    dtm = covariate_data[covariate_names.index('dtm'), :]
    geom_temp_doy = get_temperature_for_year(transform, dtm)

    #y = np.empty((n_pixels, n_output_bands), dtype=np.float32)
    timespans = np.empty((n_pixels, sequence_length), dtype=np.float32)
    x = np.empty((n_pixels, sequence_length, n_features), dtype=np.float32)
    valid_pixels_ind = np.ones(n_pixels, dtype=bool)

    @njit(parallel=True, fastmath=True)
    def _process_one_image(day_from_start, n_pixels, n_bands, 
                           x, timespans, valid_pixels_ind,
                           days_from_start, 
                           landsat_data, 
                           modis_data,
                           geom_temp_doy,
                           valid_values_mask,
                           sequence_length):
        n_dates = days_from_start.shape[0]
        for pix in prange(n_pixels):
            valid_values = valid_values_mask[:,pix]           
            pix_dfs = days_from_start[valid_values]
            ind = np.nonzero(pix_dfs < day_from_start)[0][-sequence_length:]
            dfs = pix_dfs[ind]
            if len(ind) < sequence_length:
                valid_pixels_ind[pix] = False
                continue
            for b in range(n_bands):
                x[pix, :, b] = landsat_data[b*n_dates:(b+1)*n_dates, pix][valid_values][ind]
            x[pix, :, 7] = modis_data[valid_values, pix][ind]
            ind_doys = dfs % 23
            x[pix, :, 8] = geom_temp_doy[ind_doys, pix]
            timespans[pix, :-1] = dfs[1:] - dfs[:-1]
            timespans[pix, -1] = day_from_start - dfs[-1]

    _process_one_image(day_from_start, n_pixels, n_bands,
                       x, timespans, valid_pixels_ind,
                       ds.days_from_start,
                       landsat_data, modis_data, geom_temp_doy,
                       valid_values_mask,
                       sequence_length)


    if not valid_pixels_ind.all():
        x = x[valid_pixels_ind, :, :]
        timeless = covariate_data[:, valid_pixels_ind].T
    else:
        timeless = covariate_data.T

    x[np.isnan(x)] = 0
    x[:,:,7]  = x[:,:,7]/10000
    x[:,:,8]  = x[:,:,8]/100

#%%
    xx = torch.tensor(x)#.unsqueeze(1)
    tt = torch.tensor(timespans)#.unsqueeze(1)
    batchsize = xx.size(0)
    x_timeless = torch.tensor(timeless).expand(batchsize, -1)
    y_hat = model(xx, tt, x_timeless)

    y_hat = y_hat.detach().numpy()  
    
    # 8min for full image
# %%
    import rasterio
    
    fld_out = Path('/mnt/nibble/gen_cog/arcov2/predictions')
    for b in range(n_output_bands):
        band_name = bands_prefix_out[b]
        fn = fld_out / f"{tile}_cfcv4_{year}{month:02d}_{band_name}.tif"
        prd = y_hat[:, b].reshape(utils.y_size, utils.x_size)

        profile=dict(
            driver='GTiff',
            count=1,
            dtype='float32',
            width=utils.x_size,
            height=utils.y_size,
            crs=meta[0],
            transform=meta[1]
        )

        with rasterio.open(fn, 'w', **profile) as dst:
            dst.write(prd, 1)

# %%
    import matplotlib.pyplot as plt

    for b in range(n_output_bands):
        band_name = bands_prefix_out[b]
        min_val, max_val = np.nanpercentile(y_hat[:, b], [2, 98])
        plt.imshow(y_hat[:, b].reshape(utils.y_size, utils.x_size), cmap='YlGn', vmin=min_val, vmax=max_val)
        plt.gca().set_xticks([])  # Remove x-axis ticks
        plt.gca().set_yticks([])  # Remove y-axis ticks
        plt.title(band_name)
        plt.show()

# %% 0,3,2
    plt.imshow(y_hat[:, [0,3,2]].reshape(utils.y_size, utils.x_size, 3)/1.272)
# %%
