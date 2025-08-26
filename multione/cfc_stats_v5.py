#%%
from ctypes import util
import datetime
import numpy as np
from pandas.core.dtypes import missing
from sklearn.datasets import images
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import time

from cfc_v5 import CfcLearner_v5, ArcoV2DatasetV3
import cfc_sample
from cfc_dataset import _process_one_pixel
from settings import bands_prefix_out
import utils
from utils import get_temperature_for_doy, get_temperature_for_year
from numba import njit, prange
import matplotlib.pyplot as plt
import fastgif
import rasterio
import tqdm

import torch; import intel_extension_for_pytorch as ipex
import openvino as ov

fld_out = Path('/mnt/nibble/gen_cog/arcov2/predictions')
fn_ckpt = Path('/mnt/nibble/gen_cog/arcov2/cfc-v5_e-38.ckpt')
#%%
def statistics():
    
    ds = ArcoV2DatasetV3(Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.zarr"),  
                         np.arange(2000, 2024), 12,limit=10, read_timeless=False)
    #_, vds = ds.get_train_validation_subset(0.2)
    # rndgen=np.random.default_rng(43)
    # inds = np.arange(len(ds)); rndgen.shuffle(inds)
    # valprc = 0.2; 
    #vds = Subset(ds, inds[:int(len(ds)*valprc)])
    dl = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=8)

    model = CfcLearner_v5.load_from_checkpoint(fn_ckpt, map_location='cpu', strict=True) # input_size=input_size, sequence_length=12, output_size=output_size)
    #submodel = model.model
    #submodel = submodel.eval()
    # model.freeze()
    # m = ipex.optimize(submodel, 
    #                       dtype=torch.float32, 
    #                       replace_dropout_with_identity=True,
    #                       #election = True
    #                       )
    # m.compile()
    # this model (m) crash the kernel when evaluated

    y=[]; prdy=[]
    for i, (yb, xb, tsb) in tqdm.tqdm(enumerate(dl), total=len(dl)): #tqdm.tqdm(dl): #
        # (yb, xb, tsb, ) = next(iter(dl))
        #print(i, yb.shape, xb.shape, tsb.shape, tlb.shape)

        y.append(yb)
        prdy.append(model(xb, tsb).detach())


    y = torch.cat(y, dim=0)
    prdy = torch.cat(prdy, dim=0)

    print("Statistics:")
    print(f"  - MAE: {(torch.abs(y - prdy)).mean(dim=0)}")
    print(f"  - MSE: {(torch.mean((y - prdy) ** 2, dim=0))}")
    print(f"  - R2: {(1 - torch.var(y - prdy, dim=0) / torch.var(y, dim=0))}")


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

    ds = ArcoV2DatasetV3(Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.zarr"),  years, sequence_length,limit=10, read_timeless=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

#%%
    fn_ckpt = Path('/mnt/nibble/gen_cog/arcov2/cfc-v5_e-38.ckpt')
    input_size = 9 #dataset.n_features
    output_size = 7 #dataset.n_output_bands
    sequence_length = 12
    #n_timeless_features = 17 #dataset.n_timeless_features
    model = CfcLearner_v5.load_from_checkpoint(fn_ckpt, map_location='cpu', strict=True) # input_size=input_size, sequence_length=12, output_size=output_size)
    model.freeze()

#%%
    pixel_ind = 11000700
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
    #timeless = covariate_data[:, pixel_ind]

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
    #x_timeless = torch.tensor(timeless).expand(batchsize, -1)
    y_hat = model(xx, tt)

    print(torch.nn.MSELoss()(y_hat, torch.tensor(y)).item())
    y_hat = y_hat.detach().numpy()  

#%%
    dates = ds.dates[valid_values]
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(7, 1, figsize=(10, 30))
    for b in range(7):
        ax = axs[b]
        ax.plot(dates[sequence_length:], y[:,b],'bo-', label='observed')
        ax.plot(dates[sequence_length:], y_hat[:,b], 'r.', label='predicted')
        ax.set_title(f"Band {bands_prefix_out[b]}")
        if b==0: 
            ax.legend()
    plt.show()
#%%
def test_whole_image(debug=False):
    #%%
    # https://www.intel.com/content/www/us/en/developer/articles/technical/pytorch-quantization-using-intel-neural-compressor.html
    
    tiles = ['055W_06S','015E_43N', '090W_49N']  #'055W_06S'
    year = 2020

    years = np.arange(2000, 2024)
    sequence_length = 12

    for tile in tiles:
        # tile = tiles[0]
        time0 = time.time()
        (success, error, eta), meta, valid_data, data = cfc_sample.get_tile_data(tile)    

        profile=dict(
                driver='GTiff',
                count=1,
                dtype='uint16',
                width=utils.x_size,
                height=utils.y_size,
                crs=meta[0],
                transform=meta[1],
                nodata=65535,
                blockxsize=1024, 
                blockysize=1024,    
                tiled=True,
                compress='deflate',
            )
        nodata = profile['nodata']
        
        (n_valid_pixels, inds_valid_pixels, valid_values_mask) = valid_data
        (landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy) = data

        n_bands = landsat_data.shape[0] // (23 * len(years))
        n_output_bands = 7
        n_features = 9

        ds = ArcoV2DatasetV3(None,  years, sequence_length,limit=10, read_timeless=True)
        time1=time.time()
        utils.ttprint(f'Tile {tile} loaded in {time1-time0:.0f} seconds')
        #dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    #%%
        time1=time.time()
        fn_ckpt = Path('/mnt/nibble/gen_cog/arcov2/cfc-v5_e-38.ckpt')
        input_size = 9 #dataset.n_features
        output_size = 7 #dataset.n_output_bands
        sequence_length = 12
        #n_timeless_features = 17 #dataset.n_timeless_features
        model = CfcLearner_v5.load_from_checkpoint(fn_ckpt, map_location='cpu', strict=True) # input_size=input_size, sequence_length=12, output_size=output_size)
        submodel = model.model

        # submodel.eval()
        # ov_model = None
        

        submodel = submodel.eval()
        m = ipex.optimize(submodel, 
                          dtype=torch.float32, 
                          replace_dropout_with_identity=True,
                          #election = True
                          )
        m.compile()

        
        #model.freeze()
        utils.ttprint(f'Model loaded in {time.time()-time1:.0f} seconds')
    #%%
        # predict whole image for some date
        time1 = time.time()
        sequence_length = ds.sequence_length
        
        for month in range(1, 13):
            # month=6
            time2 = time.time()
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
                timespans = timespans[valid_pixels_ind, :]
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
            #x_timeless = torch.tensor(timeless).expand(batchsize, -1)

            # if ov_model is None:
            #     ov_model = ov.convert_model(submodel, 
            #                 example_input=(xx[:10], tt[:10], x_timeless[:10]),)
           
            #y_hat = model(xx, tt, x_timeless)
           
            with torch.no_grad():
                y_hat = m(xx, tt)

            y_hat = y_hat.detach().numpy()  
    
    # 8min for full image
# %%
        
            for b in range(n_output_bands):
                # b=0
                band_name = bands_prefix_out[b]
                fn = fld_out / f"{tile}_cfcv5_{year}{month:02d}_{band_name}.tif"
                if not valid_pixels_ind.all():
                    prd = np.full(n_pixels, nodata, dtype=np.uint16)
                    prd[valid_pixels_ind] = (y_hat[:, b]*10000).astype(np.uint16)
                    prd = prd.reshape(utils.y_size, utils.x_size)
                else:
                    prd = (y_hat[:, b]*10000).astype(np.uint16).reshape(utils.y_size, utils.x_size)

                with rasterio.open(fn, 'w', **profile) as dst:
                    dst.write(prd, 1)

            utils.ttprint(f"{month}. processed in {time.time()-time2:.0f} seconds")

        utils.ttprint(f"Whole year processed in {time.time()-time1:.0f} seconds")
        utils.ttprint(f"Total time for {tile} is {time.time()-time0:.0f} seconds")

# %% Create GIF
# def create_gifs(predicted_year):
#     val_min /= 12; val_max /= 12
#     for b in range(n_output_bands):
#         for j in range(12):
#             def draw_band(band):

#                 band_name = bands_prefix_out[b]
#                 plt.imshow(predicted_year[j], cmap='YlGn', vmin=val_min[band], vmax=val_max[band])
#                 plt.gca().set_xticks([])  # Remove x-axis ticks
#                 plt.gca().set_yticks([])  # Remove y-axis ticks
#                 plt.title(band_name)
#                 plt.show()

# %% 0,3,2
    #plt.imshow(y_hat[:, [0,3,2]].reshape(utils.y_size, utils.x_size, 3)/1.272)
# %%
if __name__ == "__main__":
    test_whole_image()


# Timings
'''
[08:10:09] Loading tile 055W_06S
[08:10:36] Landsat data loaded in 27.61 seconds
[08:11:09] Processing 552 MODIS NDVI files in parallel...
[08:12:31] MODIS NDVI data loaded in 114.66 seconds
[08:12:31] Landsat + modis: 142.28 seconds
[08:12:31] Masking data ...
[08:12:44] Masked Landsat data from QA in 13.48 seconds
[08:13:10] Masked Landsat data from MODIS in 25.71 seconds
[08:13:10] Masked Landsat data in 39.19 seconds
[08:13:10] Scaling and trimming Landsat data ...
[08:13:23] Scaled Landsat data by 10000
[08:13:23] Trimmed Landsat data to 3864 rows
[08:13:23] Landsat data scaled and trimmed in 13.10 seconds
[08:13:42] Getting covariates ...
[08:14:26] Covariates loaded in 43.99 seconds
[08:14:26] Getting geom_temp_doy ...
[08:14:28] Got geom_temp_doy in 1.43 seconds
[08:14:28] Total time for loading tile 055W_06S: 259.29 seconds
[08:14:28] Tile 055W_06S loaded in 259 seconds
[08:14:28] Model loaded in 0 seconds
[08:23:23] 1. processed in 535 seconds
[08:31:51] 2. processed in 507 seconds
[08:40:21] 3. processed in 510 seconds
[08:49:06] 4. processed in 525 seconds
[08:57:51] 5. processed in 525 seconds
[09:06:37] 6. processed in 525 seconds
[09:15:03] 7. processed in 506 seconds
[09:23:49] 8. processed in 526 seconds
[09:32:37] 9. processed in 528 seconds
[09:41:23] 10. processed in 526 seconds
[09:50:07] 11. processed in 524 seconds
[09:58:37] 12. processed in 510 seconds
[09:58:37] Whole year processed in 6249 seconds
[09:58:37] Total time 6508 seconds

'''