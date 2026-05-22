#%%
import gc
import json
import time
from pathlib import Path
import numba
from numba import njit, prange
import numpy as np
import rasterio
import ray
import psutil

import data_utils as dutils
import production_utils as putils
from datetime import datetime, timedelta
from settings import IMG_SCALE_FACTOR, N_IMAGES_PER_YEAR, N_THREADS, OUTPUT_PROFILE
from settings import (
        X_SIZE, Y_SIZE, IMG_NODATA, IMG_DTYPE,
    )
#from settings import DATES_TO_PREDICT, MASTER_LOGGER_FILE, PREDICTIONS_FOLDER, SETTINGS_DICT, FILE_ENDING_OUT

# putils.init()
fld = Path('/mnt/nibble/gen_cog/arcov2/averages')
log_file = fld/'master_average.log'

years = list(range(2000, 2024))
months = list(range(1, 13))

weight_after=0.25
weight_before=0.25
bands = ['red', 'nir', 'blue', 'green', 'swir1', 'swir2', 'thermal']

numba.config.THREADING_LAYER = 'threadsafe'
print('Numba num threads:', numba.config.NUMBA_NUM_THREADS)
print('Numba threading layer:',numba.config.THREADING_LAYER)

n_cores = psutil.cpu_count(logical=False) or 0
print(f'Number of physical CPU cores: {n_cores}')

N_PROCESSES = max(1, n_cores // 8)
print(f'Number of processes to use for saving TIFF: {N_PROCESSES}')
N_THREADS = max(1, n_cores // N_PROCESSES)
print(f'Number of threads per process: {N_THREADS}')

# numba.set_num_threads(N_THREADS)
# %%

@njit(parallel=True)
def compute_average(data_input, data_output, yearmonth_weights, yearmonth_images, no_data=IMG_NODATA):
    ndates, npixels = data_input.shape
    n_weights = yearmonth_weights.shape[1]
    n_months = yearmonth_weights.shape[0]

    for i in prange(npixels):
        for ym_idx in range(n_months):
            weights = yearmonth_weights[ym_idx]
            images = yearmonth_images[ym_idx]

            weight_sum = 0.0
            val = 0.0            
            for j in range(n_weights):
                data = data_input[images[j], i]
                if (weights[j]>0) & np.isfinite(data):
                    weight_sum += weights[j]
                    val += data * weights[j]                    
            if weight_sum>0:
                data_output[ym_idx, i] = val / weight_sum
            else:
                data_output[ym_idx, i] = no_data

@ray.remote(num_cpus=N_THREADS)
def save_image(fn_output: Path, img: np.ndarray, profile: dict) -> str:
    #print(f'Saving image {fn_output}')
    with rasterio.Env(GDAL_NUM_THREADS=str(N_THREADS)):
        with rasterio.open(fn_output, 'w', **profile) as dst:  # type: ignore
            dst._set_all_scales([1. / IMG_SCALE_FACTOR])
            dst._set_all_offsets([0.0])
            dst.write(img.reshape((Y_SIZE, X_SIZE)), 1)
    return fn_output.name

#%%
def average_monthly_production(tile: str) -> None:
    # tile = '055W_06S'
    log = dutils.ProductionLogger(log_file, silent=False)
    log.log(f'AVERAGING TILE', 'START', 0, f'Starting tile {tile}')
    log.log(f'LOADING TILE DATA', 'START', 0, f'Starting loading')
    time_start_tile = time.time()
    data = dutils.load_tile_data_all(tile, log) # It will be better to load in arrays [npixels, ndates]
    del data['modis_ndvi']
    del data['ndvi']
    log.log(f'LOADING TILE DATA', 'END', time.time() - time_start_tile, f'Finished loading')

    # Get dates
    log.log(f'PREPARING WEIGHTS', 'START', 0, f'Starting preparing weights')
    time_start_weights = time.time()
    dates_start=[]
    dates_end = []
    for y in years:
        dates_start.extend([datetime(y,1,1)+timedelta(days=16*i) for i in range(N_IMAGES_PER_YEAR)])
        dates_end.extend([min(datetime(y,1,17)+timedelta(days=16*i-1), datetime(y,12,31)) for i in range(N_IMAGES_PER_YEAR)])

    weights = np.zeros((len(years) * 12, 4), dtype=np.float32)  # 
    images = np.zeros((len(years) * 12, 4), dtype=np.int32)  #

    month_days = []
    for idx, (ds, de) in enumerate(zip(dates_start, dates_end)):
        m1 = ds.month
        m2 = de.month
        if m1 == m2:
            days_in_m1 = (de - ds).days + 1
            month_days.append([idx, ds.year, m1, days_in_m1])
        else:
            days_in_m1 = (datetime(ds.year, m2, 1) - ds).days
            days_in_m2 = (de - datetime(de.year, m2, 1)).days + 1
            month_days.append([idx, ds.year, m1, days_in_m1])
            month_days.append([idx, de.year, m2, days_in_m2])

    month_days = np.array(month_days, dtype=np.int32)

    yearmonth_weights = []
    yearmonth_images = []
    idx_month = 0
    for y in years:
        for m in months:
            # y=years[0]; m=2
            weights=np.zeros(5, dtype=np.float32)
            images=np.zeros(5, dtype=np.int32)

            # images in this month
            wm = month_days[(month_days[:,1]==y) & (month_days[:,2]==m)]
            for i, row in enumerate(wm):
                weights[i] = row[3]
                images[i] = row[0]

            i = len(wm) - 1
            # image before this month
            wm = month_days[month_days[:,0]==images[0]-1]
            if (len(wm)>0):       
                i+=1         
                weights[i] = wm[0,3] * weight_before
                images[i] = wm[0,0]

            # image after this month
            wm = month_days[month_days[:,0]==images.max()+1]
            if (len(wm)>0):
                i+=1
                weights[i] = wm[0,3] * weight_after
                images[i] = wm[0,0]

            weights = weights/weights.sum()
            yearmonth_weights.append(weights)
            yearmonth_images.append(images)

    yearmonth_weights = np.array(yearmonth_weights, dtype=np.float32)
    yearmonth_images = np.array(yearmonth_images, dtype=np.int32)
    log.log(f'PREPARING WEIGHTS', 'END', time.time() - time_start_weights, f'Finished preparing weights')

    fld_out = fld / 'v1' / tile
    fld_out.mkdir(parents=True, exist_ok=True)

#%%
    time_start_bands = time.time()
    log.log(f'AVERAGING ALL BANDS', 'START', 0, f'Starting averaging for tile {tile}')
    
    # Compute averages
    for band in bands:
        log.log(f'AVERAGING BAND', 'START', 0, f'Starting band {band} for tile {tile}')
        time_start = time.time()
        data_input = data[band]
        data_output = np.zeros((yearmonth_weights.shape[0], data[band].shape[1]), dtype=np.float32)
        compute_average(data_input, data_output, yearmonth_weights, yearmonth_images, no_data=-1)        

        remotes = []
        for y in years:
            for m in months:
                
                ym_idx = (y - years[0]) * 12 + (m - 1)
                img = data_output[ym_idx]
                mask = img < 0
                img = (img * IMG_SCALE_FACTOR).astype(IMG_DTYPE)
                img[mask] = IMG_NODATA                

                fn_output = fld_out / f'{tile}_{y}{m:02d}_{band}.tif'
                profile: dict = data['profile'].copy()  # type: ignore                
                profile.update(OUTPUT_PROFILE)

                remotes.append(save_image.remote(fn_output, img, profile))
                # with rasterio.open(fn_output, 'w', **profile) as dst: # type: ignore
                #     dst._set_all_scales([1./IMG_SCALE_FACTOR])
                #     dst._set_all_offsets([0.0])
                #     dst.write(img.reshape((Y_SIZE, X_SIZE)), 1)

        while len(remotes)>0:
            done, remotes = ray.wait(remotes, num_returns=1)
            for fn in done:
                print(f'Saved image {ray.get(fn)}')

        log.log(f'AVERAGING BAND', 'END', time.time() - time_start, f'Finished band {band} for tile {tile}')        
        gc.collect()
    
    log.log(f'AVERAGING ALL BANDS', 'END', time.time() - time_start_bands, f'Finished averaging for tile {tile}')

    log.log(f'AVERAGING TILE', 'END', time.time() - time_start_tile, f'Finished tile {tile}')



    print(f'Numba threading layer used: {numba.threading_layer()}')
#%%


    

if __name__ == '__main__':
    ray.init(ignore_reinit_error=True)
    average_monthly_production('015E_43N')
    ray.shutdown()



# tiles=['055W_06S','090W_49N','015E_43N']