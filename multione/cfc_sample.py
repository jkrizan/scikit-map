#%%
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import gc

from pystac.extensions import raster
from rasterio import vrt
from requests import get

import utils
import time
import matplotlib.pyplot as plt
from pathlib import Path
import pandas

import subprocess,re,random
import xarray as xr

years = np.arange(2000, 2024)


VALID_VALUES_PERC = 0.1 #0.2
VALID_PIXELS_PERC = 0.001
VALID_PIXELS_MIN = 10

fn_log = Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.log")
fn_zarr = Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.zarr")
fn_zarr = Path(f"/mnt/nibble/gen_cog/arcov2/sample_v2.zarr")
fn_tiles = Path(f"/mnt/nibble/gen_cog/arcov2/tiles_v1.txt")

fn_log = Path(f"/mnt/nibble/gen_cog/arcov2/sample_v6.log")
fn_zarr = Path(f"/mnt/nibble/gen_cog/arcov2/sample_v6.zarr")
fn_tiles = Path(f"/mnt/nibble/gen_cog/arcov2/tiles_v6.txt")


n_years = len(years)
n_dates = n_years * utils.n_imag_per_year
n_valid_dates = int(VALID_VALUES_PERC * n_dates)
n_bands = utils.n_spect_bands

def randomize_tiles():
    """
    This needs to be run only once to get and randomize order of tiles
    """
    ls = subprocess.run('mc ls g1/prod-landsat-ard2/', shell=True, capture_output=True, text=True, check=True).stdout
    tiles = re.findall(r'.\w{3}_\w{3}', ls, flags=re.MULTILINE)
    random.shuffle(tiles)
    with open(fn_tiles, 'w') as f:
        f.writelines("\n".join(tiles))
    

#%%
def get_tile_data(landsat_tile: str, dtm_derivatives: bool = False) -> tuple[tuple[bool, str, float], tuple, tuple, tuple]:

    #global crs, transform, bounds, landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy

    utils.ttprint(f"Loading tile {landsat_tile}")
    start = start0 = time.time()
    try:        
        landsat_files = utils.get_landsat_filenames_gaia(landsat_tile, years)    
        landsat_data, crs, transform, bounds = utils.get_landsat_data(landsat_files, years)
        utils.ttprint(f"Landsat data loaded in {time.time() - start:.2f} seconds")
        start1 = time.time()
        modis_data = utils.get_modis_ndvi_data_rio(years, crs, bounds)
        utils.ttprint(f"MODIS NDVI data loaded in {time.time() - start1:.2f} seconds")  
        utils.ttprint(f"Landsat + modis: {time.time() - start:.2f} seconds")

        # masking
        utils.ttprint("Masking data ...")   
        start = time.time()
        landsat_data = utils.mask_from_qa(landsat_data, len(years)) #TODO: parallelize
        utils.ttprint(f"Masked Landsat data from QA in {time.time() - start:.2f} seconds")  
        start1 = time.time()
        landsat_data = utils.mask_from_modis(landsat_data, modis_data, len(years))  # This is heavy parallel and fast
        utils.ttprint(f"Masked Landsat data from MODIS in {time.time() - start1:.2f} seconds")
        utils.ttprint(f"Masked Landsat data in {time.time() - start:.2f} seconds")

        # scale landsat data and trim QA and NDVI
        # we don't need QA and NDVI anymore in landsat_data
        utils.ttprint("Scaling and trimming Landsat data ...")
        start = time.time()
        max_ind = utils.n_spect_bands*len(years)*utils.n_imag_per_year
        landsat_data = utils.landsat_data_trim_scale(landsat_data, max_ind, 40000)    # scale all bands with 40000    #TODO: it can be parallized !!! (numba)
        utils.ttprint(f"Landsat data scaled and trimmed in {time.time() - start:.2f} seconds")
    except Exception as e:
        eta = time.time() - start0
        utils.ttprint(f"Error occurred while processing Landsat data: {e}")
        return (False, f'Landsat, MODIS: {e}', eta), None, None, None

    # check valid pixels
    n_bands = landsat_data.shape[0] // n_dates
    modis_data = modis_data/10000  # scale modis data
    modis_data[np.isnan(modis_data)] = -1  # set nodata to -1
    
    
    utils.ttprint("Getting geom_temp_doy ...")
    start = time.time()
    dtm, lat_rows = utils.get_dtm_data(landsat_files)
    geom_temp_doy = utils.get_temperature_for_year(transform, dtm, lat_rows)
    # temp_min, temp_max = utils.temperature_min_max(dtm, lat_rows)
    utils.ttprint(f"Got geom_temp_doy in {time.time() - start:.2f} seconds")

    valid_values_mask = (modis_data >= 0)
    for b in range(n_bands):
        valid_values_mask &= np.isfinite(landsat_data[b*n_dates:(b+1)*n_dates]) # all bands should be valid
    valid_values_mask[:, ~(np.isfinite(geom_temp_doy).all(axis=0))] = False  # geom_temp_doy should be valid
    
    n_valid_values = valid_values_mask.sum(axis=0)
    inds_valid_pixels = np.where(n_valid_values >= n_valid_dates)[0]
    n_valid_pixels = inds_valid_pixels.size
    
    if int(n_valid_pixels*VALID_PIXELS_PERC) < VALID_PIXELS_MIN: # don't load if can't get at least 10 pixels
        eta = time.time() - start0
        utils.ttprint(f"Not enough valid pixels in {landsat_tile}, skipping...")
        return (False, "Not enough valid pixels", eta), (crs, transform, bounds), (n_valid_pixels, None, None), None
    
    try:
        # get covariates
        if dtm_derivatives:
            utils.ttprint("Getting covariates ...")
            start = time.time()
            covariate_data, covariate_names = utils.get_dtm_derivatives(landsat_files) 
            utils.ttprint(f"Covariates loaded in {time.time() - start:.2f} seconds")
        else:                            
            covariate_data, covariate_names = None, None

        stop = time.time()
        eta = stop - start0
        utils.ttprint(f"Total time for loading tile {landsat_tile}: {eta:.2f} seconds")
    except Exception as e:        
        eta = time.time() - start0
        utils.ttprint(f"Error occurred while processing covariates: {e}")
        return (False, f'Covariates: {e}', eta), (crs, transform, bounds), (n_valid_pixels, None, None), None

    return (
        (True, '', eta), 
        (crs, transform, bounds), 
        (n_valid_pixels, inds_valid_pixels, valid_values_mask),
        (landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy)
    )


def sample_tiles():

    tiles = fn_tiles.read_text().splitlines()
    if fn_log.exists():
        log = pandas.read_table(fn_log)
        used_tiles = log['tile'].unique()
        for t in used_tiles:
            if t in tiles:
                tiles.remove(t)
    else:
        with fn_log.open('w') as f:
            f.write("tile\tsuccess\tn_valid_pixels\tn_sampled_pixels\ttime\terror\n")

    for i, tile in enumerate(tiles):
        print()
        utils.ttprint(f"----------------------------------------------")
        utils.ttprint(f"Processing tile {i+1}/{len(tiles)}: {tile}")
        (success, error, eta), meta, valid_data, data = get_tile_data(tile, dtm_derivatives=False)
        if meta is not None:
            (crs, transform, bounds) = meta
        if valid_data is not None:
            (n_valid_pixels, inds_valid_pixels, valid_values_mask) = valid_data
        else:
            (n_valid_pixels, inds_valid_pixels, valid_values_mask) = (0, None, None)
        if not success:
            with fn_log.open('a') as f:
                f.write(f"{tile}\t{success}\t{n_valid_pixels}\t0\t{eta}\t`{error}`\n")
            continue
        else:                                    
            n_sampled_pixels = int(n_valid_pixels*VALID_PIXELS_PERC)
            inds = np.random.choice(inds_valid_pixels, size=n_sampled_pixels, replace=False) #type: ignore
            (landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy) = data #type: ignore

            msdata = modis_data[:,inds]  # type: ignore            
            gtmpdata = geom_temp_doy[:, inds]  # type: ignore
            valid_values_mask = valid_values_mask[:, inds] # type: ignore


            lsdata = np.empty((n_bands, n_dates, n_sampled_pixels), dtype=np.float32)  # type: ignore
            for b in range(n_bands):
                lsdata[b,:,:] = landsat_data[b*n_dates:(b+1)*n_dates, inds]
                        
            ds = xr.Dataset(
                {
                    "lsdata": (("band", "dates", "pixel"), lsdata),
                    "modis": (("dates","pixel"), msdata),
                    "valid_values_mask": (("dates", "pixel"), valid_values_mask),  # type: ignore
                    "geom_temp_doy": (("doy", "pixel"), gtmpdata),
                    "pixel_inds": (("pixel"), inds)
                },
                attrs={
                    "tile": tile,
                    "n_valid_pixels": n_valid_pixels,
                    "n_sampled_pixels": n_sampled_pixels,
                    "time": eta,
                }
            )

            ds.to_zarr(fn_zarr, group=tile, consolidated=False, mode='a')
            
            with fn_log.open('a') as f:
                f.write(f"{tile}\t{success}\t{n_valid_pixels}\t{n_sampled_pixels}\t{eta}\t`{error}`\n")

            del landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy

            gc.collect()

def sample_surface_water():

    import rasterio, rasterio.vrt
    import zarr
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.enums import Resampling
    from utils import get_tile_profile, ttprint, y_size, x_size
    from pathlib import Path   
    import numpy as np 

    fn_zarr = Path(f"/mnt/nibble/gen_cog/arcov2/sample_v6.zarr")
    url = "/mnt/nibble/gen_cog/arcov2/occurrence.surface.water_jrc.gswe_p_30m_s_19840101_20211231_go_epsg.4326_v20251008.tif"

    dataset: zarr.Group = zarr.open(fn_zarr, mode='r+') #type: ignore
    tiles = list(dataset.group_keys())
    ntiles = len(tiles)
    ttprint(f"Number of tiles in the dataset: {ntiles}")
    with rasterio.open(url) as src:
        water_mask_lower = src.bounds[1]
        water_mask_upper = src.bounds[3]
        ttprint(f"Water mask bounds: {src.bounds})")
        # src = rasterio.open(url)
        for i,tile in enumerate(tiles):
            # tile = tiles[0]
            ttprint(f'{i}/{ntiles}: {tile}, ', end='')
            ds: zarr.Group = dataset[tile]  #type: ignore
            array_keys = list(ds.array_keys())  #type: ignore
            if 'water_mask' in array_keys:
                ttprint("Already has water mask, skipping ...")
                continue
            ttprint("Sampling water mask ...")
            (crs, transform, bounds) = get_tile_profile(tile)     
            if bounds[1] > water_mask_upper or bounds[3] < water_mask_lower:
                ttprint("Tile outside water mask bounds, skipping ...")
                continue  

            with rasterio.vrt.WarpedVRT(src, crs=crs, resampling=Resampling.nearest) as vrt:
                window = vrt.window(*bounds)
                data = vrt.read(1, window=window, out_shape=(y_size, x_size), out_dtype=np.byte)
                ds.create_array('water_mask', data=data, chunks=(y_size, x_size)) 
            ttprint(f"{tile} done.")

                

    # Add code to download and process the water occurrence data from the URL
    with rasterio.open(url) as src:         
         profile = src.profile
         print(profile)
    pass


def statistics():
    df = pandas.read_table(fn_log)
    print(df.info())
    print(df.describe())

    print(f"Total tiles loaded: {len(df)}")
    print("Success rate (%):", df['success'].value_counts(normalize=True))
    print(f"Successfully loaded {df.success.sum()} tiles, sampled {df.n_sampled_pixels.sum()} pixels")
    print(f"Mean time to load: {df[df.success]['time'].mean():.0f} seconds, minimum: {df[df.success]['time'].min():.0f} seconds, maximum: {df[df.success]['time'].max():.0f} seconds")    


# %%
if __name__ == "__main__":
    #sample_tiles()
    sample_surface_water()