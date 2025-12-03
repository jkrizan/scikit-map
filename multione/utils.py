# Various utils 
#%%

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, List, Tuple
from numpy.typing import NDArray, ArrayLike
import gc
import rasterio as rio
import rasterio.vrt, rasterio.enums
from tqdm import tqdm
import os

from datetime import datetime
from pathlib import Path
import random
import numpy as np

from settings import n_imag_per_year, n_imag_per_year_agg, doy_start, doy_end, n_pix, x_size, y_size, gdal_opts, x_off, y_off
from settings import gaia_addrs, bands_prefix, landsat_file_ending, no_data # TMP_DIR, n_threads
from settings import mask_result_scaling, mask_band_scaling, mask_result_offset
from settings import filter_params
from settings import att_env, att_seas, future_scaling
from settings import n_spect_bands, bands_prefix_out, file_ending_out, no_data_out, month_start, month_end
try:
    from settings import s3_aliases, s3_params, s3_setup
except:
    print("s3_aliasses not imported !")
from settings import fft_th, gap_stripes_th, gap_general_th, inpaint_chunk_size, inpaint_radius, inpaint_padding
from settings import bands_scales_real
from settings import lulc_base_path, lulc_filenames, lulc_default_year, lulc_legend_filename
from settings import dtm_adresses, dtm_vars

# from processing_utils import get_SWA_weights
#from skmap import data, parallel

#%%

# Function to set up S3 aliases using MinIO Client (mc)
# This function takes access key, secret key, and a list of Gaia addresses,
# def s3_setup(access_key, secret_key, gaia_addrs) -> List[str]:
#     s3_aliases = []
#     s3_aliases = [f'g{i+1}' for i, _ in enumerate(gaia_addrs)]
#     commands = [
#         f'sudo mc alias set  g{i+1} {addr} {access_key} {secret_key} --api S3v4'
#         for i, addr in enumerate(gaia_addrs)
#     ]
#     for cmd in commands:
#         subprocess.run(cmd, shell=True, capture_output=False, text=True, check=True)
#     return s3_aliases

def setup_gaia():
    
    from settings import gaia_s3_params

    s3_aliases = s3_setup(gaia_s3_params['s3_access_key'],
                gaia_s3_params['s3_secret_key'],
                gaia_s3_params['s3_addresses'])
    
    return s3_aliases

def ttprint(*args, **kwargs):    
    print(f'[{datetime.now():%H:%M:%S}] ', end='')
    print(*args, **kwargs, flush=True)

def make_tempdir(basedir='skmap', make_subdir = True) -> Path:
    import tempfile

    tempdir = Path(TMP_DIR).joinpath(basedir)
    if make_subdir: 
        name = Path(tempfile.NamedTemporaryFile().name).name
        tempdir = tempdir.joinpath(name)
    tempdir.mkdir(parents=True, exist_ok=True)
    return tempdir

# def get_modis_ndvi_filename(year) -> str:
#     """
#     Get the filename for MODIS NDVI data based on year, start and end DOY, and band.
#     """
#     #return f'MOD13Q1.A{year}{doy_start}.{doy_end}.{band}.hdf'
#     fn = f'/vsicurl/http://192.168.49.{random.randint(30,44)}:8333/global/veg/ndvi_mod13q1.v061_swa/ndvi_mod13q1.v061_m_250m_s_{year}{doy_start[m]}_{year}{doy_end[m]}_go_sinusoidal_v1.tif'
#     return fn

# %% Min and max temperature calculated from latitude, doy, elevation
# https://opengeohub.github.io/spatial-prediction-eml/introduction-to-spatial-and-spatiotemporal-data.html#modeling-seasonal-components
# https://doi.org/10.1002/2013JD020803

def temperature_min_max(dtm, lat_rows):
    a = 30.419375
    b = -15.539232
    t_grad = 0.6
    doy1 = 18  # Days of the year
    doy2 = 200  # Days of the year
    lat_rows = lat_rows.reshape((-1, 1))  # Reshape lat_rows to be a column vector

    costeta1 = np.cos((doy1-18)*np.pi/182.5 + np.pow(2, 1-np.sign(lat_rows)) * np.pi)
    costeta2 = np.cos((doy2-18)*np.pi/182.5 + np.pow(2, 1-np.sign(lat_rows)) * np.pi)
    # wolframalpha.com: "extremes of f(t)=cos[(t-18)*pi/182.5 + pi] for t in (1, 366)"
    # for lat_rows>0, costeta min=-1 for doy=18, and max=1 for doy=200
    # for lat_rows<0, costeta min=-1 for doy=200, and max=1 for doy=18
        
    A = np.cos(lat_rows * np.pi / 180) # cosfi
    # max(A, lat=0) = 1, min(A, lat=+-90) = 0
    sin_lat = np.abs(np.sin(lat_rows * np.pi / 180))
    B1 = (1 - costeta1) * sin_lat
    B2 = (1 - costeta2) * sin_lat
    # costeta=-1 -> max(B,lat=+-90) = 2, min(B,lat=0) = 0
    # costeta=1 -> max(B) = 0, min(B) = 0

    tmpz = t_grad * dtm / 100
    x1 = a*A + b*B1 - tmpz
    x2 = a*A + b*B2 - tmpz
    # max A + min B = a - t_grad * dtm/100, lat=0, doy=200
    # min A + max B = b - t_grad * dtm/100, lat=+-90, doy=18

    return np.minimum(x1, x2), np.maximum(x1, x2)

def temperature_doy(doy, dtm, lat_rows, i):
    a = 30.419375
    b = -15.539232
    t_grad = 0.6
    lat_rows = lat_rows.reshape((-1, 1))  # Reshape lat_rows to be a column vector

    costeta = np.cos((doy-18)*np.pi/182.5 + np.pow(2, 1-np.sign(lat_rows)) * np.pi)

    # wolframalpha.com: "extremes of f(t)=cos[(t-18)*pi/182.5 + pi] for t in (1, 366)"
    # for lat_rows>0, costeta min=-1 for doy=18, and max=1 for doy=200
    # for lat_rows<0, costeta min=-1 for doy=200, and max=1 for doy=18
        
    A = np.cos(lat_rows * np.pi / 180) # cosfi
    # max(A, lat=0) = 1, min(A, lat=+-90) = 0
    sin_lat = np.abs(np.sin(lat_rows * np.pi / 180))
    B = (1 - costeta) * sin_lat
    # costeta=-1 -> max(B,lat=+-90) = 2, min(B,lat=0) = 0
    # costeta=1 -> max(B) = 0, min(B) = 0

    tmpz = t_grad * dtm / 100
    x = a*A + b*B - tmpz
    
    # max A + min B = a - t_grad * dtm/100, lat=0, doy=200
    # min A + max B = b - t_grad * dtm/100, lat=+-90, doy=18

    return x, i

def get_temperature_for_year(landsat_files, dtm, lat_rows=None):
    if type(landsat_files).__name__ == 'Affine':
        transform = landsat_files
        if lat_rows is None:
            lat_rows = rio.transform.xy(transform, np.arange(y_size), np.zeros(y_size))[1].astype(np.float32)
    else:
        if lat_rows is None:
            lat_rows = get_lat_rows(landsat_files)

    geom_temp_doy = np.empty((n_imag_per_year, n_pix), dtype=np.float32)

    from numba import njit, prange
    @njit(parallel=True, fastmath=True)
    def _temp_doy(geom_temp_doy, lat_rows, dtm):
        a = 30.419375
        b = -15.539232
        t_grad = 0.6

        for i in prange(geom_temp_doy.shape[0]):
            doy = 8 + i*16
            costeta = np.cos((doy-18)*np.pi/182.5 + np.pow(2, 1-np.sign(lat_rows)) * np.pi)
            A = np.cos(lat_rows * np.pi / 180) # cosfi
            sin_lat = np.abs(np.sin(lat_rows * np.pi / 180))
            B = (1 - costeta) * sin_lat
            tmpz = t_grad * dtm / 100
            #x = np.empty_like(tmpz)
            nrows = A.shape[0]
            aAbB = a*A+b*B
            for j in prange(nrows):                
                geom_temp_doy[i, j*nrows:(j+1)*nrows] = aAbB[j]  -tmpz[j*nrows:(j+1)*nrows] 

    _temp_doy(geom_temp_doy, lat_rows, dtm.ravel())

    return geom_temp_doy

def get_temperature_for_doy(doy, dtm, transform):
    '''
    Get the temperature for a specific day of the year (DOY).
    '''
    '''
    with rasterio.open(landsat_files[0]) as src:
        lat_rows = src.xy(np.arange(src.height), np.zeros(src.width))[1].astype(np.float32)
    return lat_rows
    '''
    lat_rows = rio.transform.xy(transform, np.arange(y_size), np.zeros(y_size))[1].astype(np.float32)
    geom_temp_doy = np.empty((n_pix,), dtype=np.float32)

    from numba import njit, prange
    @njit(parallel=True, fastmath=True)
    def _temp_doy(doy, geom_temp_doy, lat_rows, ncols):
        a = 30.419375
        b = -15.539232
        t_grad = 0.6

        costeta = np.cos((doy-18)*np.pi/182.5 + np.pow(2, 1-np.sign(lat_rows)) * np.pi)
        A = np.cos(lat_rows * np.pi / 180) # cosfi
        sin_lat = np.abs(np.sin(lat_rows * np.pi / 180))
        B = (1 - costeta) * sin_lat
        tmpz = t_grad * dtm / 100
        #x = np.empty_like(tmpz)
        nrows = A.shape[0]
        aAbB = a*A+b*B
        for j in prange(nrows):                
            geom_temp_doy[j*ncols:(j+1)*ncols] = aAbB[j]  -tmpz[j*ncols:(j+1)*ncols] 

    _temp_doy(doy, geom_temp_doy, lat_rows, x_size)
    return geom_temp_doy

    # executor = ThreadPoolExecutor(max_workers=n_threads)
    # futures = [executor.submit(temperature_doy, 8+i*16, dtm, lat_rows, i) for i in range(n_imag_per_year)]
    # for future in as_completed(futures):
    #     x, i = future.result()
    #     ttprint(f"Calculated geometric temperature for doy {i}/{n_imag_per_year}")
    #     geom_temp_doy[i] = x

    # executor.shutdown()
    # return geom_temp_doy

def get_dates_doy(years) -> Tuple[List[datetime], List[int], List[int]]:
    """
    Get the list of dates and DOY for the given years.
    """
    from datetime import datetime, timedelta

    dates_list=[]
    for y in years:
        dates = [datetime(y,1,8)+timedelta(days=16*i) for i in range(n_imag_per_year)]
        dates_list.extend(dates)        
    t = [(d - datetime(years[0],1,1)).days + 1 for d in dates_list]
    doy = list(range(8, 366, 16))*len(years)  
    return dates_list, t, doy

def get_landsat_filenames_gaia(landsat_tile, years) -> List[str]:
    landsat_files = []
    for b in bands_prefix:
        for year in years:
            for m in range(n_imag_per_year):
                landsat_files.append(f'{random.choice(gaia_addrs)}/prod-landsat-ard2/{landsat_tile}/raw/{b}.ard2_m_30m_s_{year}{doy_start[m]}_{year}{doy_end[m]}{landsat_file_ending}')

    return landsat_files

def get_landsat_filenames_local(landsat_tile, years, fld_source) -> List[str]:
    landsat_files = []
    for b in bands_prefix:
        for year in years:
            for m in range(n_imag_per_year):
                # Path(f'/mnt/nibble/gen_cog/arcov2/landsat_{landsat_tile}')
                landsat_files.append(f'{fld_source}/landsat_{landsat_tile}/{b}.ard2_m_30m_s_{year}{doy_start[m]}_{year}{doy_end[m]}{landsat_file_ending}')
    return landsat_files

def get_tile_profile(tile: str) -> Any:
    """
    Get the profile of the given tile.
    """
    fn = f'{random.choice(gaia_addrs)}/prod-landsat-ard2/{tile}/raw/{bands_prefix[0]}.ard2_m_30m_s_20010101_20010116{landsat_file_ending}'
    with rio.open(fn) as src:
        crs = src.crs
        transform = src.transform
        bounds = src.bounds

    return crs, transform, bounds

def get_landsat_data(landsat_files, years) -> Tuple[NDArray[np.float32], Any, Any, Any]:
    """
    Get the Landsat data based on year, start and end month, and band.
    """    
    from imports import skmap_bindings as sb    

    with rio.open(landsat_files[0]) as src:
        crs = src.crs
        transform = src.transform
        bounds = src.bounds

    n_years = len(years)
    n_s = n_years*n_imag_per_year
    n_s_agg = n_years*n_imag_per_year_agg
    landsat_data = np.empty((n_s*(n_spect_bands + 2), n_pix), dtype=np.float32)
    sb.readData(landsat_data, n_threads, landsat_files, range(len(landsat_files)), x_off, y_off, x_size, y_size, [1], gdal_opts, no_data, np.nan)
    # sb.readData(landsat_data, n_threads, landsat_files, [2], x_off, y_off, x_size, y_size, [1], gdal_opts, no_data, np.nan)
    # sb.readData(landsat_data, n_threads, ld, [0], x_off, y_off, x_size, y_size, [1], gdal_opts, no_data, np.nan)
    return landsat_data, crs, transform, bounds

def get_modis_ndvi_rio(modis_file, i, crs, bounds, resampling_strategy=rasterio.enums.Resampling.cubic_spline):
    '''
    ref_file = landsat_files[11]
    modis_file = modis_files[11][0]
    resampling_strategy=rasterio.enums.Resampling.bilinear
    '''
    # with rio.open(ref_file) as ref:
    #     #profile = ref.profile
    #     dst_crs = ref.crs
    #     bounds = ref.bounds
    #     # dd = ref.read(1)

    try:
        with rio.open(modis_file) as src:        
            warp_options = {
                'crs': crs,            
                'resampling': resampling_strategy
            }
            with rasterio.vrt.WarpedVRT(src, **warp_options) as vrt:
                window = vrt.window(*bounds)            
                data = vrt.read(1,window=window, out_shape=(y_size, x_size), out_dtype=np.float32, resampling=resampling_strategy)

            data[data == src.nodata] = np.nan  # Set nodata values to NaN
    except:
        return None, modis_file, i # type: ignore

    return data, modis_file, i # type: ignore

def get_modis_ndvi_data_rio(years, crs, bounds, resampling_strategy=rasterio.enums.Resampling.cubic_spline) -> NDArray[np.float32]:

    modis_files = []
    for year in years:
        for m in range(n_imag_per_year):
            modis_files.append(f'/vsicurl/{random.choice(gaia_addrs)}/global/veg/ndvi_mod13q1.v061_swa/ndvi_mod13q1.v061_m_250m_s_{year}{doy_start[m]}_{year}{doy_end[m]}_go_sinusoidal_v1.tif')


    n_years = len(years)
    n_s = n_years*n_imag_per_year
    modis_data = np.empty((n_s, n_pix), dtype=np.float32)
    executor = ProcessPoolExecutor(max_workers=n_threads)    
    futures = [executor.submit(get_modis_ndvi_rio, modis_files[i], i, crs, bounds, resampling_strategy)
               for i in range(len(modis_files))]
    
    ttprint(f"Processing {len(modis_files)} MODIS NDVI files in parallel...")
    for future in as_completed(futures):
        data, modis_file, i  = future.result() # type: ignore
        if data is None:
            ttprint(f"Failed to process {modis_file}")
            #futures.append(executor.submit(get_modis_ndvi_rio, ref_file, modis_file, resampling_strategy))
        else:
            # ttprint(f"Processed {modis_file} successfully")
            modis_data[i, :] = data.ravel()

    executor.shutdown()
    return modis_data
    

def get_modis_ndvi_data(landsat_files, years, resampling_strategy='GRA_Bilinear') -> NDArray[np.float32]:
    '''
    Old code from Davide
    '''
    #landsat_files = get_landsat_filenames(landsat_tile, years)    
    from imports import warp_tile

    modis_files = []
    for year in years:
        for m in range(n_imag_per_year):
            modis_files.append(f'/vsicurl/{random.choice(gaia_addrs)}/global/veg/ndvi_mod13q1.v061_swa/ndvi_mod13q1.v061_m_250m_s_{year}{doy_start[m]}_{year}{doy_end[m]}_go_sinusoidal_v1.tif')

    n_years = len(years)
    n_s = n_years*n_imag_per_year
    modis_data = np.empty((n_s, n_pix), dtype=np.float32)
    executor = ProcessPoolExecutor(max_workers=n_threads)
    futures = [executor.submit(warp_tile, landsat_files[i], modis_files[i], n_pix, resampling_strategy)
        for i in range(len(modis_files))]
    for i, future in enumerate(futures):
        modis_data[i, :] = future.result()

    executor.shutdown()
    return modis_data

def get_lulc_data(landsat_files, years, class_level) -> NDArray[np.int8]:

    with rasterio.open(landsat_files[0]) as src:
        profile = src.profile
        bounds = src.bounds

    lulc_data = np.empty((len(years), n_pix), dtype=np.int8)
    for i, year in enumerate(years):
        fn = lulc_filenames[year]
        with rasterio.open(fn) as src:
            warp_options = {
                'crs': profile['crs'],            
                'resampling': rasterio.enums.Resampling.nearest,  # Use nearest neighbor for categorical data
            }
            with rasterio.vrt.WarpedVRT(src, **warp_options) as vrt:
                window = vrt.window(*bounds)            
                data = vrt.read(1,window=window, out_shape=(profile['height'], profile['width']), 
                                out_dtype=np.int8, resampling=rasterio.enums.Resampling.nearest)

            data[data == src.nodata] = np.nan  # Set nodata values to NaN

    return lulc_data

def get_lat_rows(landsat_files):
    with rasterio.open(landsat_files[0]) as src:
        lat_rows = src.xy(np.arange(src.height), np.zeros(src.width))[1].astype(np.float32)
    return lat_rows

def get_dtm_data(landsat_files) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:

    with rio.open(landsat_files[0]) as src:
        profile = src.profile
        bounds = src.bounds
        lat_rows = src.xy(np.arange(profile['height']), np.zeros(profile['width']))[1].astype(np.float32)
    
    fn = gaia_addrs[0] + dtm_vars['dtmv3']  # Use the first address for the DTM file
    with rio.open(fn) as src:
        window = src.window(*bounds)
        data = src.read(1, window=window, out_shape=(profile['height'], profile['width']),
                        out_dtype=np.float32, resampling=rasterio.enums.Resampling.bilinear)
        data[data == src.nodata] = np.nan        
        
    return data, lat_rows

def get_data(fn, bounds, i):
    with rio.open(fn) as src:
        window = src.window(*bounds)
        data = src.read(1, window=window, out_shape=(y_size, x_size), 
                        out_dtype=np.float32, resampling=rasterio.enums.Resampling.bilinear)
        data[data == src.nodata] = np.nan
    return data, i

def get_dtm_derivatives(landsat_files):

    with rio.open(landsat_files[0]) as src:        
        bounds = src.bounds

    futures = []
    dtm_derivatives_names=[]
    executor = ProcessPoolExecutor(max_workers=n_threads)    
    i=0
    for dtmvar in dtm_vars.keys():
        if dtmvar.startswith('dtmv'):
            continue
        futures.append(executor.submit(get_data, f"{np.random.choice(gaia_addrs)}{dtm_vars[dtmvar]}", bounds, i))
        dtm_derivatives_names.append(dtmvar)
        i += 1

    dtm_derivatives = np.empty((len(futures), n_pix), dtype=np.float32)
    for future in as_completed(futures):
        data, i = future.result()
        dtm_derivatives[i, :] = data.ravel()
        #print(f"Read data for {dtm_derivatives_names[i]}")

    executor.shutdown()
    return dtm_derivatives, dtm_derivatives_names

def get_dtm_covariates(landsat_files):
    dtm_derivatives, dtm_derivatives_names = get_dtm_derivatives(landsat_files)
    dtm, lat_rows = get_dtm_data(landsat_files)
    temp_min, temp_max = temperature_min_max(dtm, lat_rows)

    covariate_data = np.concatenate((dtm_derivatives, dtm.reshape(1,-1), temp_min.reshape(1,-1), temp_max.reshape(1,-1)), axis=0)
    covariate_names = dtm_derivatives_names + ['dtm', 'temp_min', 'temp_max']

    del dtm_derivatives, dtm, lat_rows, temp_min, temp_max

    return covariate_data, covariate_names

def mask_from_qa(landsat_data: NDArray[np.float32], n_years:int) -> NDArray[np.float32]:
    from imports import skmap_bindings as sb
    
    n_s = n_years*n_imag_per_year
    #range_qa = range(n_s*(n_spect_bands), n_s*(n_spect_bands+1))
    
    #landsat_mask = landsat_data[range_qa, :]
    # Try removing snow, check 16d_intervals.xlsx for the QA info and scaling
    # 14 = additional cloud buffer over land
    # 3 = cloud
    # 6 = snow
    #                         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17   
    gap_mask_keep_buffer   = [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    gap_mask_remove_buffer = [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]

    # @FIXME this thing is basically not parallel
    ind_start = n_s*n_spect_bands
    landsat_mask = np.empty((n_s, n_pix), dtype=np.float32)
    sb.fillArray(landsat_mask, n_threads, 1.)

    def mask_one_image(i) -> None:
        n_cloud_pix = np.sum((landsat_data[ind_start + i,:] == 3))
        n_buff_pix = np.sum((landsat_data[ind_start + i,:] == 14))  # avoid division by 0
        gap_mask = gap_mask_remove_buffer if (n_cloud_pix > n_buff_pix) else gap_mask_keep_buffer

        mask_zeros = np.nonzero(np.logical_not(gap_mask))[0]
        ind = np.isin(landsat_data[ind_start + i,:], mask_zeros) # kind='table' is faster but only for integer arrays
        landsat_mask[i,ind] = 0.

    executor = ThreadPoolExecutor(max_workers=n_threads)
    futures = [executor.submit(mask_one_image, i) for i in range(n_s)]
    for future in as_completed(futures):
        future.result()

    executor.shutdown()
    '''
    for k in range(0,18):
        sb.swapRowsValues(landsat_mask, n_threads, [i], k, gap_mask[k])
    '''
    # This is a workaround for the above commented code, which is not parallel
    #mask_ones = np.nonzero(gap_mask)[0]
    # mask_zeros = np.nonzero(np.logical_not(gap_mask))[0]
    # ind = np.isin(landsat_data[ind_start + i,:], mask_zeros) # kind='table' is faster but only for integer arrays
    # landsat_mask[i,ind] = 0.
        

    for i in range(n_spect_bands):
        sb.maskData(landsat_data, n_threads, range(n_s*i, n_s*(i+1)), landsat_mask, 1., np.nan)

    del landsat_mask
    gc.collect()

    return landsat_data

def mask_from_qa_parallel(landsat_data: NDArray[np.float32], n_years:int) -> NDArray[np.float32]:
    # Use parallel processing to mask Landsat data from QA
    from imports import skmap_bindings as sb    

    from concurrent.futures import ThreadPoolExecutor

    n_s = n_years*n_imag_per_year

    #landsat_mask = landsat_data[range_qa, :]
    # Try removing snow, check 16d_intervals.xlsx for the QA info and scaling
    # 14 = additional cloud buffer over land
    # 3 = cloud
    # 6 = snow
    #                         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17   
    gap_mask_keep_buffer   = [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    gap_mask_remove_buffer = [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]

    # @FIXME this thing is basically not parallel
    ind_start = n_s*n_spect_bands
    landsat_mask = np.empty((n_s, n_pix), dtype=np.float32)
    sb.fillArray(landsat_mask, n_threads, 1.)

    def process_mask_row(i: int) -> None:
        n_cloud_pix = np.sum((landsat_data[ind_start + i,:] == 3))
        n_buff_pix = np.sum((landsat_data[ind_start + i,:] == 14))  # avoid division by 0
        gap_mask = gap_mask_remove_buffer if (n_cloud_pix > n_buff_pix) else gap_mask_keep_buffer

        mask_zeros = np.nonzero(np.logical_not(gap_mask))[0]
        ind = np.isin(landsat_data[ind_start + i,:], mask_zeros) # kind='table' is faster but only for integer arrays
        landsat_mask[i,ind] = 0.
        
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(process_mask_row, i) for i in range(n_s)]
        for future in futures:
            future.result()
        
    for i in range(n_spect_bands):
        sb.maskData(landsat_data, n_threads, range(n_s*i, n_s*(i+1)), landsat_mask, 1., np.nan)

    del landsat_mask
    gc.collect()

    return landsat_data

def mask_from_modis(landsat_data: NDArray[np.float32], modis_data:NDArray[np.float32], n_years:int) -> NDArray[np.float32]:
    """
    Create a mask from the QA band of Landsat data.
    """
    from imports import skmap_bindings as sb    

    n_s = n_years*n_imag_per_year
    n_s_agg = n_years*n_imag_per_year_agg

    range_nir = range(n_s*1, n_s*2)
    range_red = range(n_s*0, n_s*1)
    range_ndvi = range(n_s*(n_spect_bands+1), n_s*(n_spect_bands+2))
    range_qa = range(n_s*(n_spect_bands), n_s*(n_spect_bands+1))

    diff_th, count_th = filter_params(n_years)
    
    sb.computeNormalizedDifference(landsat_data, n_threads,
                                range_nir, range_red, range_ndvi,
                                mask_band_scaling, mask_band_scaling, mask_result_scaling, 
                                mask_result_offset, [-mask_result_scaling, mask_result_scaling])
    landsat_NDVI_masked = np.empty((n_s, n_pix), dtype=np.float32)
    sb.extractArrayRows(landsat_data, n_threads, landsat_NDVI_masked, range_ndvi)
    landsat_NDVI_masked_t = np.empty((n_pix, n_s), dtype=np.float32)
    sb.transposeArray(landsat_NDVI_masked, n_threads, landsat_NDVI_masked_t)

    modis_data_t = np.empty((n_pix, n_s), dtype=np.float32)
    sb.transposeArray(modis_data, n_threads, modis_data_t)

    modis_ndvi_mask_t = np.empty((n_pix, n_s), dtype=np.float32)
    modis_ndvi_mask = np.empty((n_s, n_pix), dtype=np.float32)
    sb.maskDifference(landsat_NDVI_masked_t, n_threads, diff_th, count_th, modis_data_t, modis_ndvi_mask_t)
    sb.transposeArray(modis_ndvi_mask_t, n_threads, modis_ndvi_mask)

    for i in range(n_spect_bands):
        sb.maskData(landsat_data, n_threads, range(n_s*i, n_s*(i+1)), modis_ndvi_mask, 1., np.nan)

    del modis_data
    del modis_data_t
    del landsat_NDVI_masked
    del landsat_NDVI_masked_t
    del modis_ndvi_mask
    del modis_ndvi_mask_t
    gc.collect()

    return landsat_data

def inpaint_stripes(landsat_data: NDArray[np.float32], n_years:int) -> None:
    """
    Inpaint stripes in Landsat data.
    """
    from imports import skmap_bindings as sb

    n_s = n_years*n_imag_per_year

    sample_idxs_band0, row_starts_band0, row_ends_band0, col_starts_band0, col_ends_band0, fill_true_erase_false_band0 = [], [], [], [], [], []
    sample_idxs, row_starts, row_ends, col_starts, col_ends, fill_true_erase_false = [], [], [], [], [], []
    for i in range(n_s):
        tmp_image = landsat_data[i].reshape(x_size, y_size)
        row_starts_tmp, row_ends_tmp, col_starts_tmp, col_ends_tmp, fill_true_erase_false_tmp, _, _, _ = process_image_in_chunks(tmp_image, inpaint_chunk_size, gap_stripes_th, gap_general_th, fft_th)
        sample_idxs_band0 += [i for x in row_starts_tmp]
        row_starts_band0 += row_starts_tmp
        row_ends_band0 += row_ends_tmp
        col_starts_band0 += col_starts_tmp
        col_ends_band0 += col_ends_tmp
        fill_true_erase_false_band0 += fill_true_erase_false_tmp

    for i in range(n_spect_bands):
        sample_idxs_band = [x + n_s*i for x in sample_idxs_band0]
        sample_idxs += sample_idxs_band
        row_starts += row_starts_band0
        row_ends += row_ends_band0
        col_starts += col_starts_band0
        col_ends += col_ends_band0
        fill_true_erase_false += fill_true_erase_false_band0
        
    row_starts_fill, row_starts_erase = [], []
    row_ends_fill, row_ends_erase = [], []
    col_starts_fill, col_starts_erase = [], []
    col_ends_fill, col_ends_erase = [], []
    sample_idxs_fill, sample_idxs_erase = [], []
    for si, rs, re, cs, ce, fe in zip(sample_idxs, row_starts, row_ends, col_starts, col_ends, fill_true_erase_false):
        if fe:
            row_starts_fill.append(rs)
            row_ends_fill.append(re)
            col_starts_fill.append(cs)
            col_ends_fill.append(ce)
            sample_idxs_fill.append(si)
        else:
            row_starts_erase.append(rs)
            row_ends_erase.append(re)
            col_starts_erase.append(cs)
            col_ends_erase.append(ce)
            sample_idxs_erase.append(si)
            

    sb.inpaintChunks(landsat_data, n_threads, inpaint_radius, inpaint_padding, x_size, y_size, sample_idxs_fill, row_starts_fill, row_ends_fill, col_starts_fill, col_ends_fill)
    sb.eraseChunks(landsat_data, n_threads, x_size, y_size, sample_idxs_erase, row_starts_erase, row_ends_erase, col_starts_erase, col_ends_erase)


def bands_aggregation(landsat_data: NDArray[np.float32], n_years:int) -> NDArray[np.float32]:
    """
    Aggregate Landsat data bands.
    """
    from imports import skmap_bindings as sb

    n_s = n_years*n_imag_per_year
    n_s_agg = n_years*n_imag_per_year_agg

    landsat_bands = np.empty((n_s*n_spect_bands, n_pix), dtype=np.float32)
    sb.extractArrayRows(landsat_data, n_threads, landsat_bands, range(0, n_s*n_spect_bands))
    del landsat_data
    gc.collect()

    landsat_bands_t = np.empty((n_pix, n_s*n_spect_bands), dtype=np.float32)
    sb.transposeArray(landsat_bands, n_threads, landsat_bands_t)
    del landsat_bands
    gc.collect()

    landsat_bands_agg_t = np.empty((n_pix, n_s_agg*n_spect_bands), dtype=np.float32)
    agg_pattern = []
    for i in range(n_spect_bands):
        for j in range(n_years):
            base_idx = n_s*i + j*n_imag_per_year
            agg_pattern.append([base_idx+0,base_idx+1])
            agg_pattern.append([base_idx+2,base_idx+3])
            agg_pattern.append([base_idx+4,base_idx+5])
            agg_pattern.append([base_idx+6,base_idx+7])
            agg_pattern.append([base_idx+8,base_idx+9])
            agg_pattern.append([base_idx+10,base_idx+11])
            agg_pattern.append([base_idx+11,base_idx+12])
            agg_pattern.append([base_idx+13,base_idx+14])
            agg_pattern.append([base_idx+15,base_idx+16])
            agg_pattern.append([base_idx+17,base_idx+18])
            agg_pattern.append([base_idx+19,base_idx+20])
            agg_pattern.append([base_idx+21,base_idx+22])
            
    sb.nanMeanAggregatePattern(landsat_bands_t, n_threads, landsat_bands_agg_t, agg_pattern)
    del landsat_bands_t
    gc.collect()

    return landsat_bands_agg_t

def swa_reconstructing(landsat_bands_agg_t: NDArray[np.float32], n_years:int) -> NDArray[np.float32]:
    """
    Reconstruct SWA from aggregated Landsat bands.
    """
    from imports import skmap_bindings as sb

    w_0_agg = 1.0
    n_s_agg = n_years*n_imag_per_year_agg

    w_p_agg = (get_SWA_weights(att_env, att_seas, n_imag_per_year_agg, n_s_agg)[1:][::-1]).astype(np.float32)
    w_f_agg = (get_SWA_weights(att_env, att_seas, n_imag_per_year_agg, n_s_agg)[1:]).astype(np.float32)*future_scaling
    
    landsat_bands_rec_t = np.empty((n_pix, n_s_agg*n_spect_bands), dtype=np.float32)

    for b in range(n_spect_bands):
        sb.applyTsirf(landsat_bands_agg_t, n_threads, landsat_bands_rec_t, n_s_agg, b*n_s_agg, b*n_s_agg, w_0_agg, w_p_agg, w_f_agg, True)
    del landsat_bands_agg_t
    gc.collect()

    return landsat_bands_rec_t

def process_image_in_chunks(image, chunk_size, gap_stripes_th, gap_general_th, fft_th):
    mask = np.isnan(image)
    height, width = image.shape
    n_chunk_height = int(np.floor(height/chunk_size))
    n_chunk_width = int(np.floor(width/chunk_size))
    gap_fraq = np.zeros((n_chunk_height, n_chunk_width))
    fft_score = np.zeros((n_chunk_height, n_chunk_width))
    rec_flag = np.zeros((n_chunk_height, n_chunk_width))
    #output_image = image.copy()
    row_starts, row_ends, col_starts, col_ends, fill_true_erase_false = [], [], [], [], []
    # Loop through the image by chunks
    for i in range(0, n_chunk_height):
        for j in range(0, n_chunk_width):
            # @FIXME check is also theretically the location of patial frequencies in different share chunks is the same 
            if i != (n_chunk_height-1):
                row_start, row_end = (i * chunk_size, (i+1) * chunk_size)
            else:
                row_start, row_end = (i * chunk_size, height)
            if j != (n_chunk_width-1):
                col_start, col_end = (j * chunk_size, (j+1) * chunk_size)
            else:
                col_start, col_end = (j * chunk_size, width)
            image_chunk = image[row_start:row_end, col_start:col_end]
            mask_chunk = mask[row_start:row_end, col_start:col_end]
            gap_count_chunk = np.sum(mask_chunk)
            gap_fraq[i, j] = gap_count_chunk/(row_end-row_start)/(col_end-col_start)
            if gap_fraq[i, j] < gap_general_th:
                row_starts += [row_start]
                row_ends += [row_end]
                col_starts += [col_start]
                col_ends += [col_end]
                fill_true_erase_false += [True]
                rec_flag[i,j] = 1
            else:
                image_filled = np.nan_to_num(image_chunk, nan=0)
                image_filled = image_filled[0:chunk_size,0:chunk_size].copy()
                # image_filled /= max(np.max(image_filled),1)
                image_filled[image_filled!=0] = 1
                ft = np.fft.ifftshift(image_filled)
                ft = np.fft.fft2(ft, norm='ortho')
                ft = np.fft.fftshift(ft)
                ft[48:80,48:80] = 0
                fft_score[i, j] = np.max(np.abs(ft))
                if fft_score[i, j] > fft_th:
                    row_starts += [row_start]
                    row_ends += [row_end]
                    col_starts += [col_start]
                    col_ends += [col_end]
                    if gap_fraq[i, j] < gap_stripes_th:
                        fill_true_erase_false += [True]
                        rec_flag[i,j] = 1
                    else:
                        fill_true_erase_false += [False]
                        rec_flag[i,j] = -1
                    
    return row_starts, row_ends, col_starts, col_ends, fill_true_erase_false, gap_fraq, fft_score, rec_flag


def save_landsat_bands(landsat_bands_rec_t: NDArray[np.float32], landsat_tile: str, years: List[int], landsat_files: List[str]) -> None:    
    """
    Save reconstructed Landsat bands to disk.
    """
    from imports import skmap_bindings as sb


    n_years = len(years)
    n_s_agg = n_years*n_imag_per_year_agg

    out_data = np.empty((n_s_agg*n_spect_bands, n_pix), dtype=np.float32)
    sb.transposeArray(landsat_bands_rec_t, n_threads, out_data)
    del landsat_bands_rec_t
    gc.collect()

    
    out_dir = f'/tmp/{landsat_tile}'
    os.makedirs(out_dir, exist_ok = True)
    
    compression_command = f"gdal_translate -a_nodata {no_data_out} -co COMPRESS=deflate -co PREDICTOR=2 -co TILED=TRUE -co BLOCKXSIZE=2048 -co BLOCKYSIZE=2048"
    out_files = []
    for band in bands_prefix_out:
        for year in years:
            for m in range(n_imag_per_year_agg):
                out_files.append(f'{band}.ard2_m_30m_s_{year}{month_start[m]}_{year}{month_end[m]}{file_ending_out}')

    s3_out = [f'{random.choice(s3_aliases)}/{s3_params["s3_prefix"]}/{landsat_tile}' for _ in range(len(out_files))]
    sb.writeUInt16Data(out_data, n_threads, gdal_opts, landsat_files[0:len(out_files)], out_dir, out_files, range(len(out_files)),
                x_off, y_off, x_size, y_size, no_data_out, compression_command, s3_out)
    os.rmdir(out_dir)
    print(f"Check gaia at {s3_out[0]}")

def landsat_data_trim_scale(landsat_data, max_ind:int, scale: float) -> NDArray[np.float32]:
    """
    Trim and scale Landsat data.
    """    
    from numba import njit,prange

    @njit(parallel=True, fastmath=True)
    def scale_part(landsat_data, max_ind, scale):
        for i in prange(max_ind):
            landsat_data[i,:] =  landsat_data[i,:] / scale

    scale_part(landsat_data, max_ind, np.float32(scale))
    ttprint(f"Scaled Landsat data by {scale}")

    landsat_data.resize((max_ind, n_pix), refcheck=False)
    ttprint(f"Trimmed Landsat data to {landsat_data.shape[0]} rows")

    return landsat_data  # Return only the trimmed data

# %%
def show_image_landsat(landsat_data: NDArray[np.float32], years:List, year:int, img_in_year:int, band:int) -> None:
    """
    Show Landsat image for a specific band.
    """
    # year=2001; img_in_year=10; band=7
    # band 7=qa; band 8=ndvi
    import matplotlib.pyplot as plt
    
    n_years = len(years)
    n_s = n_years*n_imag_per_year    
    ind_band = n_s*band + (year - years[0])*n_s + img_in_year

    data = landsat_data[ind_band, :].reshape((y_size, x_size))
    plt.imshow(data)
    plt.title(f'Band {band} for year {year} image {img_in_year}')
    plt.colorbar()
    plt.show()  

def show_image_modis(modis_data: NDArray[np.float32], years:List, year:int, img_in_year:int) -> None:
    """
    Show MODIS image for a specific year and image in year.
    """
    # year=2001; img_in_year=10
    import matplotlib.pyplot as plt
            
    ind_band = (year - years[0])*n_imag_per_year + img_in_year

    data = modis_data[ind_band, :].reshape((y_size, x_size))
    plt.imshow(data)
    plt.title(f'MODIS NDVI for year {year} image {img_in_year}')
    plt.colorbar()
    plt.show()



def load_from_zarr_parallel(filename):
    """
    Load data from a Zarr file in parallel.
    """
    # filename = f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}.zarr'
    import zarr
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor    
    
    def read_block(zarr_array, i):
        return i, zarr_array.blocks[i]
    
    ret = dict()
    root = zarr.open_group(filename, mode='r')
    for key in root.array_keys():
        # key = 'modis_data'; key='years'
        ttprint(f'Loading {key} from {filename}')
        zarr_array: zarr.Array = root[key] # type: ignore
        np_array = np.empty(zarr_array.shape, dtype=zarr_array.dtype) 
        

        # TODO: load per chunks, not rows
        
        if zarr_array.ndim == 2:
            nrows = zarr_array.shape[0]
            nrows_per_block = zarr_array.chunks[0] if zarr_array.chunks is not None else nrows
            nblocks = int(np.ceil(nrows / nrows_per_block))
            with ThreadPoolExecutor(max_workers= 2*n_threads) as executor:
                futures = [executor.submit(read_block, zarr_array, i) for i in range(nblocks)]
                for future in as_completed(futures):
                    i, block = future.result()
                    np_array[i*nrows_per_block: min(nrows,(i+1)*nrows_per_block), :] = block
        elif zarr_array.ndim == 1:
            np_array[:] = zarr_array[:]
        else:
            raise ValueError(f"Unsupported array dimension: {zarr_array.ndim} for key {key}")   
        
        ret[key] = np_array

    return ret

def show_timeseries(landsat_data: NDArray[np.float32], modis_data: NDArray[np.float32], 
                    years:List[int], row: int, col: int, bands_to_show=None) -> None:
    """
    Show time series of Landsat data for a specific band and image in year.
    """
    # row=100; col=200
    import matplotlib.pyplot as plt

    bands_prefix_real = bands_prefix[:n_spect_bands] + ['qa','ndvi']

    ind_pix = row*x_size + col
    n_years = len(years)
    n_s = n_years*n_imag_per_year
    if bands_to_show is None:        
        bands_choose = np.array([0, 1, 2, 3, 4, 5, 6, 8] ) # Choose bands to show    
    else:
        bands_choose = np.array(bands_to_show)
    ind_bands = n_s*bands_choose
    scales = np.array(bands_scales_real)[bands_choose]
    #(years[0] - years[0])*n_s + img_in_year

    ldata = np.empty((len(ind_bands), n_s), dtype=np.float32)
    for i, (ind_band, scale) in enumerate(zip(ind_bands, scales)):
        ldata[i, :] = landsat_data[ind_band:ind_band+n_s, ind_pix] / scale

    mdata = modis_data[:, ind_pix] / 10000

    plt.figure(figsize=(12, 6))
    for band, lts in zip(bands_choose,ldata):
        plt.plot(lts, label=f'Band {bands_prefix_real[band]}', marker='o', linestyle='')
    plt.plot(mdata, label='MODIS NDVI', marker='', linestyle='-')
    plt.legend()

    plt.title(f'Time series for pixel [{row}, {col}]')
    plt.show()