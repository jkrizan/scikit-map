# %%

from datetime import datetime
from pathlib import Path
import time
import numpy as np
import rasterio
import rasterio.transform
import rasterio.enums
import rasterio.vrt
import ray
from typing import Any
import numba
from numba import njit, prange
from production_logger import ProductionLogger


from settings import (
    IMG_SCALE_FACTOR,
    TMP_FOLDER,
    YEARS,
    PRODUCTION_FOLDER,
    DATA_TYPE,
    MODIS_RESAMPLING_STRATEGY,
    RND_GEN,
    FILE_ENDING_OUT,
    GAIA_ADDRS, 
    DTM_FILE,
    X_SIZE,
    Y_SIZE,
    N_PIXELS,
    N_IMAGES_PER_YEAR,
    LANDSAT_FILE_ENDING,
    LANDSAT_NODATA,
    MODIS_NODATA,
    MODIS_SCALE_FACTOR,
    DOY_START, 
    DOY_END, 
    BANDS_DICT,
    LANDSAT_SCALE_FACTOR,
    FILTER_DIFF_TH,
    FILTER_COUNT_TH,
    N_NEEDED_VALID_DATES,
    OUTPUT_PROFILE
)

# %%
def get_landsat_filenames_gaia(
    landsat_tile: str, years: list[int], band: str
) -> list[str]:
    """Generate the Gaia S3 URL for a specific Landsat tile, year, and band."""
    landsat_files = []
    for year in years:
        for m in range(N_IMAGES_PER_YEAR):
            landsat_files.append(
                f"{RND_GEN.choice(GAIA_ADDRS)}/prod-landsat-ard2/{landsat_tile}/raw/{BANDS_DICT[band]}.ard2_m_30m_s_{year}{DOY_START[m]}_{year}{DOY_END[m]}{LANDSAT_FILE_ENDING}"
            )
    return landsat_files


def get_lat_rows(landsat_file):
    with rasterio.open(landsat_file) as src:
        lat_rows = src.xy(np.arange(src.height), np.zeros(src.width))[1].astype(
            np.float32
        )
    return lat_rows

def get_temperature_for_year(profile, dtm, lat_rows):
    geom_temp_doy = np.empty((N_IMAGES_PER_YEAR, N_PIXELS), dtype=DATA_TYPE)

    from numba import njit, prange

    @njit(parallel=True, fastmath=True)
    def _temp_doy(geom_temp_doy, lat_rows, dtm):
        a = 30.419375
        b = -15.539232
        t_grad = 0.6

        for i in prange(geom_temp_doy.shape[0]):
            doy = 8 + i * 16
            costeta = np.cos(
                (doy - 18) * np.pi / 182.5 + np.pow(2, 1 - np.sign(lat_rows)) * np.pi
            )
            A = np.cos(lat_rows * np.pi / 180)  # cosfi
            sin_lat = np.abs(np.sin(lat_rows * np.pi / 180))
            B = (1 - costeta) * sin_lat
            tmpz = t_grad * dtm / 100
            # x = np.empty_like(tmpz)
            nrows = A.shape[0]
            aAbB = a * A + b * B
            for j in prange(nrows):
                geom_temp_doy[i, j * nrows : (j + 1) * nrows] = (
                    aAbB[j] - tmpz[j * nrows : (j + 1) * nrows]
                )

    _temp_doy(geom_temp_doy, lat_rows, dtm.ravel())

    return geom_temp_doy


def get_temperature_for_doy(doy, dtm, transform):
    """
    Get the temperature for a specific day of the year (DOY).
    """
    """
    with rasterio.open(landsat_files[0]) as src:
        lat_rows = src.xy(np.arange(src.height), np.zeros(src.width))[1].astype(np.float32)
    return lat_rows
    """
    lat_rows = rasterio.transform.xy(transform, np.arange(Y_SIZE), np.zeros(Y_SIZE))[
        1
    ].astype(np.float32)
    geom_temp_doy = np.empty((N_PIXELS,), dtype=np.float32)

    from numba import njit, prange

    @njit(parallel=True, fastmath=True)
    def _temp_doy(doy, geom_temp_doy, lat_rows, ncols):
        a = 30.419375
        b = -15.539232
        t_grad = 0.6

        costeta = np.cos(
            (doy - 18) * np.pi / 182.5 + np.pow(2, 1 - np.sign(lat_rows)) * np.pi
        )
        A = np.cos(lat_rows * np.pi / 180)  # cosfi
        sin_lat = np.abs(np.sin(lat_rows * np.pi / 180))
        B = (1 - costeta) * sin_lat
        tmpz = t_grad * dtm / 100
        # x = np.empty_like(tmpz)
        nrows = A.shape[0]
        aAbB = a * A + b * B
        for j in prange(nrows):
            geom_temp_doy[j * ncols : (j + 1) * ncols] = (
                aAbB[j] - tmpz[j * ncols : (j + 1) * ncols]
            )

    _temp_doy(doy, geom_temp_doy, lat_rows, X_SIZE)
    return geom_temp_doy


@ray.remote
def _read_landsat_file(
    i: int, landsat_file: str, nodata: np.uint16
) -> tuple[int, np.ndarray, np.ndarray]:
    try:
        with rasterio.open(landsat_file) as src:
            landsat_data = src.read(1).flatten()
            mask = landsat_data != nodata
        return i, landsat_data, mask, None
    except Exception as e:
        landsat_data = np.full((N_PIXELS,), nodata, dtype=np.uint16)
        mask = np.zeros((N_PIXELS,), dtype=bool)
        return i, landsat_data, mask, str(e)


@numba.njit(fastmath=True, parallel=True)
def _inplace_bitwise_and(mask1: np.ndarray, mask2: np.ndarray):
    """In-place bitwise AND operation between two boolean arrays.
    result is written to mask1
    """
    for i in prange(mask1.shape[0]):
        mask1[i] = mask1[i] & mask2[i]


def get_landsat_tile_data(
    landsat_tile: str, years: list[int], bands: list[str]
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    """Reads specified bands from a Landsat tile and returns them as a NumPy array.

    Args:
        landsat_tile (str): Path to the Landsat tile file.
        years (list[int]): List of years to read.
        bands (list[str]): List of band names to read.

    Returns:
        np.ndarray: Array containing the data from the specified bands.
    """
    profile = None
    data = {}
    landsat_mask = None
    err = []
    for b in bands:
        landsat_files = get_landsat_filenames_gaia(landsat_tile, years, b)
        if profile is None:
            for lf in landsat_files:
                try:
                    with rasterio.open(lf) as src:
                        profile = src.profile
                except Exception as e:
                    pass
                    #err.append((str(e), lf))
                if profile is not None:
                    break
        n_files = len(landsat_files)
        landsat_data = np.empty((n_files, N_PIXELS), np.uint16)
        if landsat_mask is None:
            landsat_mask = np.ones((n_files, N_PIXELS), dtype=bool)
        remotes = [
            _read_landsat_file.remote(i, landsat_file, np.uint16(LANDSAT_NODATA))
            for i, landsat_file in enumerate(landsat_files)
        ]
        while len(remotes) > 0:
            finished, remotes = ray.wait(
                remotes,  # timeout=7.0
            )
            for i, data_i, mask, errimage in ray.get(finished):
                landsat_data[i, :] = data_i
                _inplace_bitwise_and(landsat_mask[i, :], mask)
                if errimage is not None:
                    err.append((errimage, landsat_files[i]))
        data[b] = landsat_data
    data["mask"] = landsat_mask  # type: ignore

    return data, profile, err  # type: ignore


def get_dtm_tile_data(profile: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Reads the DTM data for a tile and returns it as a NumPy array.

    Args:
        profile (dict[str, Any]): Profile information for the tile.

    Returns:
        np.ndarray: Array containing the DTM data.
    """
    bounds = rasterio.transform.array_bounds(X_SIZE, Y_SIZE, profile["transform"])
    with rasterio.open(DTM_FILE) as src:
        window = src.window(*bounds)
        data = src.read(
            1,
            window=window,
            out_shape=(X_SIZE, Y_SIZE),
            out_dtype=np.float32,
            resampling=rasterio.enums.Resampling.bilinear,
        )
        data[data == src.nodata] = np.nan
        lat_rows = src.xy(np.arange(profile["height"]), np.zeros(profile["width"]))[
            1
        ].astype(np.float32)

    return data, lat_rows


@ray.remote
def _read_modis_file(
    i: int, modis_file: str, crs: str, bounds: Any
) -> tuple[int, np.ndarray]:
    with rasterio.open(modis_file) as src:
        warp_options = {
            "resampling": MODIS_RESAMPLING_STRATEGY,
            "crs": crs,
        }
        with rasterio.vrt.WarpedVRT(src, **warp_options) as vrt:
            window = vrt.window(*bounds)
            # Can't read with WarpedVRT directly into np.float16, so read as float32 and convert later
            data = vrt.read(
                1,
                window=window,
                out_shape=(Y_SIZE, X_SIZE),
                out_dtype=np.float32,
                resampling=MODIS_RESAMPLING_STRATEGY,
            ).flatten()
            data[data == MODIS_NODATA] = np.nan
            data = (data * MODIS_SCALE_FACTOR).astype(DATA_TYPE)
    return i, data


def get_modis_tile_data(profile: dict[str, Any], years: list[int]) -> np.ndarray:
    """Reads specified bands from a MODIS tile and returns them as a NumPy array.

    Args:
        profile (dict[str, Any]): Profile information for the MODIS tile.
        years (list[int]): List of years to read.
        bands (list[str]): List of band names to read.

    Returns:
        np.ndarray: Array containing the data from the specified bands.
    """
    modis_files = []
    for year in years:
        for m in range(N_IMAGES_PER_YEAR):
            modis_files.append(
                f"/vsicurl/{RND_GEN.choice(GAIA_ADDRS)}/global/veg/ndvi_mod13q1.v061_swa/ndvi_mod13q1.v061_m_250m_s_{year}{DOY_START[m]}_{year}{DOY_END[m]}_go_sinusoidal_v1.tif"
            )

    bnds = rasterio.transform.array_bounds(X_SIZE, Y_SIZE, profile["transform"])

    n_files = len(modis_files)
    modis_data = np.empty((n_files, N_PIXELS), dtype=DATA_TYPE)
    remotes = [
        _read_modis_file.remote(i, modis_file, profile["crs"], bnds)
        for i, modis_file in enumerate(modis_files)
    ]
    while len(remotes) > 0:
        finished, remotes = ray.wait(
            remotes,  # timeout=7.0
        )
        for i, data in ray.get(finished):
            modis_data[i, :] = data

    return modis_data


@ray.remote
def _mask_image_from_qa(i: int, qa_data: np.ndarray) -> tuple[int, np.ndarray]:
    #                         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
    gap_mask_keep_buffer = [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    gap_mask_remove_buffer = [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]

    n_cloud_pix = np.sum((qa_data == 3))
    n_buff_pix = np.sum((qa_data == 14))
    gap_mask = (
        gap_mask_remove_buffer if (n_cloud_pix > n_buff_pix) else gap_mask_keep_buffer
    )

    mask = np.ones_like(qa_data, dtype=bool)
    mask_ones = np.nonzero(gap_mask)[0]
    ind = np.isin(qa_data, mask_ones)
    mask[ind] = False

    return i, mask


def mask_from_qa(landsat_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Creates a mask from the QA band of the Landsat data.

    Args:
        data (dict[str, np.ndarray]): Dictionary containing the Landsat data.

    Returns:
        np.ndarray: Boolean mask indicating valid pixels.
    """
    qa = landsat_data.get("qa")
    if qa is None or not isinstance(qa, np.ndarray):
        raise ValueError("QA band not found in data dictionary.")
    mask = landsat_data.get("mask")
    if mask is None:
        raise ValueError("Mask not found in data dictionary.")

    # Try removing snow, check 16d_intervals.xlsx for the QA info and scaling
    # 14 = additional cloud buffer over land
    # 3 = cloud
    # 6 = snow
    #                          0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
    # gap_mask_keep_buffer   = [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    # gap_mask_remove_buffer = [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]

    # qa_mask = np.zeros_like(qa, dtype=bool)
    remotes = [_mask_image_from_qa.remote(i, qa[i, :]) for i in range(qa.shape[0])]

    while len(remotes) > 0:
        finished, remotes = ray.wait(
            remotes,  # timeout=7.0
        )
        for i, mask_i in ray.get(finished):
            _inplace_bitwise_and(mask[i, :], mask_i)

    landsat_data["mask"] = mask
    return landsat_data


@numba.njit(fastmath=True, parallel=True)
def _cast_and_scale(
    mask: np.ndarray, src: np.ndarray, dest: np.ndarray, scale: DATA_TYPE
):
    n_images, n_pixels = src.shape
    for i in prange(n_images):
        for j in range(n_pixels):
            if mask[i, j]:
                dest[i, j] = src[i, j] * scale
            else:
                dest[i, j] = np.nan


def cast_and_scale_landsat(landsat_data: dict[str, np.ndarray]):
    """Casts and scales the specified band of the Landsat data.

    Args:
        data (dict[str, np.ndarray]): Dictionary containing the Landsat data.
        band (str): Band name to cast and scale.
        scale (float): Scale factor.

    Returns:
        np.ndarray: Scaled band data.
    """
    for b in landsat_data.keys():
        if (b in BANDS_DICT.keys()) and (b != "qa"):
            tmp = np.empty_like(landsat_data[b], dtype=DATA_TYPE)
            _cast_and_scale(
                landsat_data["mask"], landsat_data[b], tmp, LANDSAT_SCALE_FACTOR
            )
            landsat_data[b] = tmp

    return landsat_data


@numba.njit(fastmath=True, parallel=True)
def _compute_ndvi(red: np.ndarray, nir: np.ndarray, ndvi: np.ndarray, mask: np.ndarray):
    n_images, n_pixels = red.shape
    for i in numba.prange(n_images):
        for j in range(n_pixels):
            if mask[i, j]:
                denom = nir[i, j] + red[i, j]
                if denom != 0:
                    ndvi[i, j] = (nir[i, j] - red[i, j]) / denom
                else:
                    ndvi[i, j] = np.nan
                    mask[i, j] = False
            else:
                ndvi[i, j] = np.nan
                mask[i, j] = False


def compute_ndvi(landsat_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Computes the NDVI from the Landsat data.

    Args:
        data (dict[str, np.ndarray]): Dictionary containing the Landsat data.

    Returns:
        np.ndarray: NDVI values.
    """
    red = landsat_data.get("red")
    nir = landsat_data.get("nir")
    mask = landsat_data.get("mask")
    if red is None or nir is None or mask is None:
        raise ValueError("Red, NIR, or mask band not found in data dictionary.")

    # sb.computeNormalizedDifference(landsat_data, n_threads,
    #                             range_nir, range_red, range_ndvi,
    #                             mask_band_scaling, mask_band_scaling, mask_result_scaling,
    #                             mask_result_offset, [-mask_result_scaling, mask_result_scaling])

    ndvi = np.empty_like(red, dtype=DATA_TYPE)
    _compute_ndvi(red, nir, ndvi, mask)  # changes ndvi and mask in place
    landsat_data["ndvi"] = ndvi
    return landsat_data


@njit(fastmath=True, parallel=True)
def _compute_fapar(
    ndvi: np.ndarray,
    mask: np.ndarray,
    fapar: np.ndarray,
    ndvi_min,
    ndvi_max,
    fpar_min,
    fpar_max,
):
    scale = (fpar_max - fpar_min) / (ndvi_max - ndvi_min)
    n_images, n_pixels = ndvi.shape
    for i in prange(n_images):
        for j in range(n_pixels):
            if mask[i, j]:
                if ndvi[i, j] >= 0:
                    fapar[i, j] = (ndvi[i, j] - ndvi_min) * scale + fpar_min
                    if fapar[i, j] < 0:
                        fapar[i, j] = 0.0
                    elif fapar[i, j] > 1:
                        fapar[i, j] = 1.0
                else:
                    fapar[i, j] = np.nan
                    mask[i, j] = False
            else:
                fapar[i, j] = np.nan


def compute_fapar(landsat_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Computes the FAPAR from the Landsat NDVI data.

    Args:
        data (dict[str, np.ndarray]): Dictionary containing the Landsat data.

    Returns:
        np.ndarray: FAPAR values.
    """
    ndvi_min = 0.03
    ndvi_max = 0.96
    fpar_min = 0.001
    fpar_max = 0.95

    ndvi = landsat_data.get("ndvi")
    mask = landsat_data.get("mask")
    if ndvi is None or mask is None:
        raise ValueError("NDVI band or mask not found in data dictionary.")

    fapar = np.empty_like(ndvi)
    _compute_fapar(
        ndvi, mask, fapar, ndvi_min, ndvi_max, fpar_min, fpar_max
    )  # changes fapar and mask in place
    landsat_data["fapar"] = fapar

    return landsat_data


@numba.njit(fastmath=True, parallel=True)
def _mask_landsat_byndvi(
    landsat_ndvi: np.ndarray, modis_ndvi: np.ndarray, diff_th: float, count_th: int
):
    n_images, n_pixels = landsat_ndvi.shape
    for j in prange(n_pixels):
        diff_count = 0
        for i in range(n_images):
            diff = np.abs(landsat_ndvi[i, j] - modis_ndvi[i, j])
            if diff > diff_th:
                diff_count += 1
        if diff_count >= count_th:
            for i in range(n_images):
                landsat_ndvi[i, j] = np.nan


def mask_landsat_byndvi(landsat_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Applies the Landsat mask to the NDVI data.

    Args:
        ndvi (np.ndarray): NDVI values.
        landsat_mask (np.ndarray): Landsat mask.

    Returns:
        np.ndarray: Masked NDVI values.
    """

    landsat_ndvi = landsat_data.get("ndvi")
    modis_ndvi = landsat_data.get("modis_ndvi")
    if landsat_ndvi is None or modis_ndvi is None:
        raise ValueError("Landsat NDVI or MODIS NDVI not found in data dictionary.")

    _mask_landsat_byndvi(landsat_ndvi, modis_ndvi, FILTER_DIFF_TH, FILTER_COUNT_TH)
    landsat_data["ndvi"] = landsat_ndvi
    landsat_data["mask"] = ~np.isnan(landsat_ndvi)
    return landsat_data


# %%
#########################################################################################################


def load_tile_data(
    tile: str, log: ProductionLogger | None, years: list[int] = YEARS
) -> dict[str, np.ndarray]:
    """Loads the Landsat tile data for the specified years and bands.

    Args:
        landsat_tile (str): The Landsat tile identifier.
        years (list[int]): The list of years to load data for.
        bands (list[str]): The list of bands to load.

    Returns:
        dict[str, np.ndarray]: A dictionary containing the loaded Landsat data.
    """
    if log is None:
        log = ProductionLogger(TMP_FOLDER / f"{tile}.log", silent=False)

    bands = ["red", "nir", "qa"]
    time0 = time.time()

    # Load Landsat data
    time1 = time.time()
    try:
        landsat_data, profile, errs = get_landsat_tile_data(tile, years, bands)
        landsat_data["profile"] = profile   # type: ignore
    except Exception as e:
        log.log("LOAD_LANDSAT_TILE_DATA", "FAILURE", time.time() - time1, str(e))
        raise e
    if errs:
        for err_msg, err_file in errs:
            log.log("LOAD_LANDSAT_TILE_DATA_FILE_ERROR", "FAILURE", 0.0, f"{err_msg} in file {err_file}")
    log.log("LOAD_LANDSAT_TILE_DATA", "SUCCESS", time.time() - time1)

    # Load MODIS NDVI data
    time1 = time.time()
    try:
        landsat_data["modis_ndvi"] = get_modis_tile_data(profile, YEARS)
    except Exception as e:
        log.log("LOAD_MODIS_NDVI_DATA", "FAILURE", time.time() - time1, str(e))
        raise e
    log.log("LOAD_MODIS_NDVI_DATA", "SUCCESS", time.time() - time1)

    # Masking with QA
    time1 = time.time()
    try:
        landsat_data = mask_from_qa(landsat_data)
        del landsat_data["qa"]  # Remove QA band to save memory
    except Exception as e:
        log.log("MASK_FROM_QA", "FAILURE", time.time() - time1, str(e))
        raise e
    log.log("MASK_FROM_QA", "SUCCESS", time.time() - time1)

    # Cast and scale Landsat data
    time1 = time.time()
    try:
        landsat_data = cast_and_scale_landsat(landsat_data)
    except Exception as e:
        log.log("CAST_AND_SCALE_LANDSAT", "FAILURE", time.time() - time1, str(e))
        raise e
    log.log("CAST_AND_SCALE_LANDSAT", "SUCCESS", time.time() - time1)

    # Compute NDVI from landsat data
    time1 = time.time()
    try:
        landsat_data = compute_ndvi(landsat_data)
        del landsat_data["red"]
        del landsat_data["nir"]
    except Exception as e:
        log.log("COMPUTE_NDVI", "FAILURE", time.time() - time1, str(e))
        raise e
    log.log("COMPUTE_NDVI", "SUCCESS", time.time() - time1)

    # Apply MODIS-based mask to Landsat NDVI
    time1 = time.time()
    try:
        landsat_data = mask_landsat_byndvi(landsat_data)
    except Exception as e:
        log.log(
            "MASK_LANDSAT_BYNDVI",
            "FAILURE",
            time.time() - time1,
            str(e),
        )
        raise e
    log.log("MASK_LANDSAT_BYNDVI", "SUCCESS", time.time() - time1)

    # Compute FAPAR from Landsat NDVI
    time1 = time.time()
    try:
        compute_fapar(landsat_data)
        del landsat_data["ndvi"]
    except Exception as e:
        log.log("COMPUTE_FAPAR", "FAILURE", time.time() - time1, str(e))
        raise e
    log.log("COMPUTE_FAPAR", "SUCCESS", time.time() - time1)

    # Load DTM and compute geom_temp_doy
    time1 = time.time()
    try:
        dtm, lat_rows = get_dtm_tile_data(profile)
        geom_temp_doy = get_temperature_for_year(profile, dtm, lat_rows)
        del dtm, lat_rows
        landsat_data["geom_temp_doy"] = geom_temp_doy
    except Exception as e:
        log.log(
            "LOAD_DTM_AND_COMPUTE_GEOM_TEMP_DOY", "FAILURE", time.time() - time1, str(e)
        )
        raise e
    log.log("LOAD_DTM_AND_COMPUTE_GEOM_TEMP_DOY", "SUCCESS", time.time() - time1)

    # Masking valid values
    time1 = time.time()
    try:
        valid_values_mask = landsat_data["modis_ndvi"] > 0
        _inplace_bitwise_and(valid_values_mask, landsat_data["mask"])
        _inplace_bitwise_and(
            valid_values_mask, np.isfinite(landsat_data["geom_temp_doy"]).all(axis=0)
        )

        n_valid_values = np.sum(valid_values_mask, axis=0)
        mask_valid_pixels = n_valid_values >= N_NEEDED_VALID_DATES

        landsat_data["mask_valid_pixels"] = mask_valid_pixels
        landsat_data["valid_values_mask"] = valid_values_mask
        del landsat_data["mask"]
    except Exception as e:
        log.log("MASK_VALID_VALUES", "FAILURE", time.time() - time1, str(e))
        raise e
    log.log("MASK_VALID_VALUES", "SUCCESS", time.time() - time1)

    log.log("TOTAL_TILE_DATA_LOADING", "SUCCESS", time.time() - time0)

    return landsat_data

@njit(parallel=True, fastmath=True, cache=True)
def _process_one_image_fpar(day_from_start, n_pixels,  
                    x, timeless, timespans, direction, valid_pixels_ind,
                    days_from_start, 
                    fapar_data, 
                    modis_data,
                    geom_temp_doy,
                    valid_values_mask,
                    sequence_length) -> None:    

    ind_doys_all = np.arange(len(days_from_start)) % 23 # precompute all orders of images in year
    for pix in prange(n_pixels):
        # pix = 13696144; sequence_length = MODEL_SEQUENCE_LENGTH; fapar_data = fapar; modis_data = modis_ndvi; 
        valid_values = valid_values_mask[:,pix]           
        pix_dfs = days_from_start[valid_values]
        ind = np.nonzero(pix_dfs < day_from_start)[0][-sequence_length:]
        
        if len(ind) < sequence_length: # not enough valid dates for forward direction
            # let's try backward
            ind = np.nonzero(pix_dfs > day_from_start)[0][:sequence_length]
            if len(ind) < sequence_length:  # still not enough valid dates
                valid_pixels_ind[pix] = False
                continue
            else:
                direction[pix] = -1  # backward
                ind = ind[::-1]
        else:
            direction[pix] = 1  # forward

        valid_pixels_ind[pix] = True
        inds_global = np.nonzero(valid_values)[0][ind]
        dfs = pix_dfs[ind]
        x[pix, :, 0] = fapar_data[inds_global, pix]
        x[pix, :, 1] = modis_data[inds_global, pix]        
        x[pix, :, 2] = geom_temp_doy[ind_doys_all[inds_global], pix]

        timespans[pix, :-1] = dfs[1:] - dfs[:-1]
        timespans[pix, -1] = day_from_start - dfs[-1]
                                        
        # gtemp at prediction date
        last_dfs = dfs[-1]
        last_gtemp = geom_temp_doy[ind_doys_all[inds_global[-1]], pix]
        if direction[pix] == 1:
            next_dfs_mask = days_from_start > day_from_start
        else:
            next_dfs_mask = days_from_start < day_from_start

        if np.any(next_dfs_mask):
            next_dfs_ind = np.nonzero(next_dfs_mask)[0][0] if direction[pix] == 1 else np.nonzero(next_dfs_mask)[0][-1]
            next_dfs = days_from_start[next_dfs_ind]
            next_gtemp = geom_temp_doy[ind_doys_all[next_dfs_ind], pix]
            timeless[pix, 2] = last_gtemp + (next_gtemp - last_gtemp) * (day_from_start - last_dfs) / (next_dfs - last_dfs)  # gtemp at prediction date
        else:
            timeless[pix, 2] = last_gtemp    
    return

def prepare_data_for_date(
    data: dict[str, np.ndarray], predict_date: datetime):
    """Prepares the data for a specific date index.
    Args:
        landsat_data (dict[str, np.ndarray]): The Landsat data dictionary. 
        date (datetime): The date to prepare data for.
    Returns:
        dict[str, np.ndarray]: A dictionary containing the data for the specified date.
    """
    from settings import (
        N_PIXELS, MODEL_SEQUENCE_LENGTH, MODEL_INPUT_SIZE, MODEL_TIMELESS_SIZE, DATA_TYPE,
        DAYS_FROM_START)

    fapar = data["fapar"]
    modis_ndvi = data["modis_ndvi"]
    geom_temp_doy = data["geom_temp_doy"]
    valid_values_mask = data["valid_values_mask"]

    timespans = np.empty((N_PIXELS, MODEL_SEQUENCE_LENGTH), dtype=DATA_TYPE)
    x = np.empty((N_PIXELS, MODEL_SEQUENCE_LENGTH, MODEL_INPUT_SIZE), dtype=DATA_TYPE)
    timeless = np.empty((N_PIXELS, MODEL_TIMELESS_SIZE), dtype=DATA_TYPE)
    valid_pixels_mask = np.empty(N_PIXELS, dtype=bool)
    direction = np.zeros(N_PIXELS, dtype=np.int8) 

    day_from_start = (predict_date - datetime(YEARS[0], 1, 1)).days
    days_from_start = np.array(DAYS_FROM_START)

    _process_one_image_fpar(day_from_start, N_PIXELS,
                x, timeless, timespans, direction, valid_pixels_mask,
                days_from_start,
                fapar, modis_ndvi, geom_temp_doy,
                valid_values_mask,
                MODEL_SEQUENCE_LENGTH)

    timeless[:,0] = geom_temp_doy.min(axis=0)   # gtemp_min
    timeless[:,1] = geom_temp_doy.max(axis=0)   # gtemp_max

    if not valid_pixels_mask.all():
        x = x[valid_pixels_mask, :, :]
        timeless = timeless[valid_pixels_mask, :]
        timespans = timespans[valid_pixels_mask, :]
    timespans /= 366.0

    return x, timeless, timespans, valid_pixels_mask

def save_predictions(predictions, profile, valid_pixels_mask, fn_output):
    """Saves the predictions to a GeoTIFF file.
    Args:
        predictions (np.ndarray): The predictions to save.
        profile (dict): The metadata profile for the output file.
        valid_pixels_mask (np.ndarray): Boolean mask indicating valid pixels.
        fn_output (Path): The output file path.
    """
    from settings import (
        X_SIZE, Y_SIZE, IMG_NODATA, IMG_DTYPE,
    )
    
    predictions = (predictions*IMG_SCALE_FACTOR).astype(IMG_DTYPE)
    if not valid_pixels_mask.all():
        img = np.full((Y_SIZE * X_SIZE), IMG_NODATA, dtype=IMG_DTYPE)
        img[valid_pixels_mask] = predictions[:, 0]    # if there is more outputs then this needs to be adjusted
    else:
        img = predictions

    img = img.reshape((Y_SIZE, X_SIZE))

    profile = profile.copy()
    profile.update(OUTPUT_PROFILE)

    with rasterio.open(fn_output, 'w', **profile) as dst: # type: ignore
        dst._set_all_scales([1./IMG_SCALE_FACTOR])
        dst._set_all_offsets([0.0])
        dst.write(img, 1)

    # %%

def load_tile_data_all(tile: str, 
                       log: ProductionLogger | None = None,
                       years: list[int] = YEARS, 
                       bands: list[str] = list(BANDS_DICT.keys())) -> dict[str, np.ndarray]:
    """Loads the Landsat tile data for the specified years and bands.

    Args:
        landsat_tile (str): The Landsat tile identifier.
        years (list[int]): The list of years to load data for.
        bands (list[str]): The list of bands to load.

    Returns:
        dict[str, np.ndarray]: A dictionary containing the loaded Landsat data.
    """

    if log is None:
        log = ProductionLogger(TMP_FOLDER / f"{tile}.log", silent=False)

    time0 = time.time()

    # Load Landsat data
    time1 = time.time()
    try:
        landsat_data, profile, errs = get_landsat_tile_data(tile, years, bands)
        landsat_data["profile"] = profile   # type: ignore
    except Exception as e:
        log.log("LOAD_LANDSAT_TILE_DATA", "FAILURE", time.time() - time1, str(e))
        raise e
    if errs:
        for err_msg, err_file in errs:
            log.log("LOAD_LANDSAT_TILE_DATA_FILE_ERROR", "FAILURE", 0.0, f"{err_msg} in file {err_file}")
    log.log("LOAD_LANDSAT_TILE_DATA", "SUCCESS", time.time() - time1)

    # Load MODIS NDVI data
    time1 = time.time()
    try:
        landsat_data["modis_ndvi"] = get_modis_tile_data(profile, YEARS)
    except Exception as e:
        log.log("LOAD_MODIS_NDVI_DATA", "FAILURE", time.time() - time1, str(e))
        raise e
    log.log("LOAD_MODIS_NDVI_DATA", "SUCCESS", time.time() - time1)

    # Masking with QA
    time1 = time.time()
    try:
        landsat_data = mask_from_qa(landsat_data)
        del landsat_data["qa"]  # Remove QA band to save memory
    except Exception as e:
        log.log("MASK_FROM_QA", "FAILURE", time.time() - time1, str(e))
        raise e
    log.log("MASK_FROM_QA", "SUCCESS", time.time() - time1)

    print(f"TOTAL_TILE_DATA_LOADING: SUCCESS in {time.time() - time0} seconds")

    # Cast and scale Landsat data
    time1 = time.time()
    try:
        landsat_data = cast_and_scale_landsat(landsat_data)
    except Exception as e:
        log.log("CAST_AND_SCALE_LANDSAT", "FAILURE", time.time() - time1, str(e))
        raise e
    log.log("CAST_AND_SCALE_LANDSAT", "SUCCESS", time.time() - time1)

     # Compute NDVI from landsat data
    time1 = time.time()
    try:
        landsat_data = compute_ndvi(landsat_data)
    except Exception as e:
        log.log("COMPUTE_NDVI", "FAILURE", time.time() - time1, str(e))
        raise e
    log.log("COMPUTE_NDVI", "SUCCESS", time.time() - time1)


    # Apply MODIS-based mask to Landsat NDVI
    time1 = time.time()
    try:
        landsat_data = mask_landsat_byndvi(landsat_data)
    except Exception as e:
        log.log(
            "MASK_LANDSAT_BYNDVI",
            "FAILURE",
            time.time() - time1,
            str(e),
        )
        raise e
    log.log("MASK_LANDSAT_BYNDVI", "SUCCESS", time.time() - time1)

    # Masking valid values
    time1 = time.time()
    try:
        valid_values_mask = landsat_data["modis_ndvi"] > 0
        _inplace_bitwise_and(valid_values_mask, landsat_data["mask"])

        n_valid_values = np.sum(valid_values_mask, axis=0)
        mask_valid_pixels = n_valid_values >= N_NEEDED_VALID_DATES

        landsat_data["mask_valid_pixels"] = mask_valid_pixels
        landsat_data["valid_values_mask"] = valid_values_mask
        del landsat_data["mask"]
    except Exception as e:
        log.log("MASK_VALID_VALUES", "FAILURE", time.time() - time1, str(e))
        raise e
    log.log("MASK_VALID_VALUES", "SUCCESS", time.time() - time1)

    log.log("TOTAL_TILE_DATA_LOADING", "SUCCESS", time.time() - time0)

    return landsat_data

def test_get_all_data():
    import utils

    tiles = ["015E_43N"]  # ,'090W_49N', '055W_06S']

    bands = ["red", "nir", "qa"]
    time0 = time.time()
    for tile in tiles:
        # tile = tiles[0]
        print(f"Processing tile: {tile}")
        print("Reading Landsat data...")
        time1 = time.time()
        landsat_data, profile, err = get_landsat_tile_data(tile, YEARS, bands)
        print(f"Data for {tile}: {landsat_data['red'].shape}")
        print(f"Profile: {profile}")
        print(f"Time taken: {time.time() - time1} seconds")
        print("--------------------------------")

        """
        # ----- OLD
        landsat_files = utils.get_landsat_filenames_gaia(tile, YEARS)    
        landsat_data_old, crs, transform, bounds = utils.get_landsat_data(landsat_files, YEARS)
        n_dates = utils.n_imag_per_year * len(YEARS)
        red_band = 0
        # ----COMPARE----------------------        
        red_old = landsat_data_old[red_band*n_dates:(red_band+1)*n_dates]
        red_new = landsat_data['red']
        print(f"Red band old shape: {red_old.shape}, new shape: {red_new.shape}")
        red_old[np.isnan(red_old)] = 0
        print(f"Red band old vs new difference: {np.sum((red_old - red_new)**2)}")
        print(f"Red band old vs new equal: {np.array_equal(red_old, red_new)}") # PASS!
        print('--------------------------------')
        nir_old = landsat_data_old[1*n_dates:(1+1)*n_dates]
        nir_new = landsat_data['nir']
        print(f"NIR band old shape: {nir_old.shape}, new shape: {nir_new.shape}")
        nir_old[np.isnan(nir_old)] = 0
        print(f"NIR band old vs new difference: {np.sum((nir_old - nir_new)**2)}")
        print(f"NIR band old vs new equal: {np.array_equal(nir_old, nir_new)}") # PASS!
        print('================================')
        # --------------------------------
        """

        print("Reading MODIS data...")
        time1 = time.time()
        landsat_data["modis_ndvi"] = get_modis_tile_data(profile, YEARS)
        print(f"MODIS Data for {tile}: {landsat_data['modis_ndvi'].shape}")
        print(f"Time taken: {time.time() - time1} seconds")
        print("================================")

        """
        # OLD CODE ##############
        modis_data_old = utils.get_modis_ndvi_data_rio(YEARS, crs, bounds)
        # -- COMPARE
        modis_data_old[np.isnan(modis_data_old)] = 0
        modis_data[np.isnan(modis_data)] = 0
        print(f"MODIS old vs new difference: {np.sum((modis_data_old - modis_data)**2)}")
        # MODIS old vs new difference: 1.1235362080697087e-06   # PASS!
        # print(f"MODIS old vs new equal: {np.array_equal(modis_data_old, modis_data)}") # PASS!
        print('================================')
        # -----------------------
        ########################
        """

        time1 = time.time()
        print("Creating mask from QA band...")
        landsat_data = mask_from_qa(landsat_data)
        print(f"Time taken: {time.time() - time1} seconds")
        del landsat_data["qa"]  # Remove QA band to save memory
        print("--------------------------------")

        """
        # OLD CODE
        landsat_data_old = utils.mask_from_qa(landsat_data_old, len(YEARS)) 
        landsat_mask_old = ~np.isnan(landsat_data_old[:n_dates])  # Get mask from first band (red)
        # -- COMPARE
        i=100
        mask_new = landsat_data['mask']
        print(f"Mask old vs new difference: {np.sum((landsat_mask_old[i] != mask_new[i]))}")
        print(f"Mask old sum: {np.sum(landsat_mask_old[i])}, new sum: {np.sum(mask_new[i])}, difference: {np.sum(landsat_mask_old[i]) - np.sum(mask_new[i])}")
        print(f"Mask old vs new equal: {np.array_equal(landsat_mask_old[i], mask_new[i])}") # WRONG!
        ind = landsat_mask_old[i] != mask_new[i]
        qas = landsat_data['qa'][i, ind]
        # Difference is only in qf = 2 (water), 12 (Additional cloud proximity over water)
        # mask_new masks out 2,12, while old mask does not
        print(f"QA values where masks differ: {np.unique(qas)}")

        mn = mask_new[i].reshape(4004,4004).astype(np.uint8); ms = landsat_mask_old[i].reshape(4004,4004).astype(np.uint8)
        mq = np.zeros_like(mn); mq.flat[ind] =qas
        #print(f"Mask old sum: {np.sum(ms)}, new sum: {np.sum(mn)}")
        import matplotlib.pyplot as plt
        plt.imshow(mn - ms, vmin=-1, vmax=1); plt.colorbar()
        plt.imshow(mq); plt.colorbar() 
        print('There is difference in masks due to QA values 2 and 12 (water).')
        print('================================')
        # -----------------------
        """

        print("Casting and scaling Landsat data...")
        time1 = time.time()
        landsat_data = cast_and_scale_landsat(landsat_data)
        print(f"Time taken: {time.time() - time1} seconds")
        print("--------------------------------")

        print("Computing NDVI...")
        time1 = time.time()
        landsat_data = compute_ndvi(
            landsat_data
        )  # PROVJERITI DA LI SE DOBIJU ISTI REZULTATI KAO PRIJE !!!
        del landsat_data["red"]
        del landsat_data["nir"]
        print(f"Time taken: {time.time() - time1} seconds")

        print("Applying MODIS-based mask to Landsat NDVI...")
        time1 = time.time()
        landsat_data = mask_landsat_byndvi(landsat_data)
        print(f"Time taken: {time.time() - time1} seconds")
        print("================================")

        """
        # OLD CODE ##################
        landsat_data_old = utils.mask_from_modis(landsat_data_old, modis_data_old, len(YEARS)) 
        landsat_mask_old = ~np.isnan(landsat_data_old[:n_dates])  # Get mask from first band (red)
        # -- COMPARE ..............
        # ----There is some difference here in masks, cannot explain why yet ----
        """

        print("Computing FAPAR ...")
        time1 = time.time()
        compute_fapar(landsat_data)
        del landsat_data["ndvi"]
        print(f"Time taken: {time.time() - time1} seconds")
        print("--------------------------------")

        print("Geometric temperature calculation...")
        time1 = time.time()
        dtm, lat_rows = get_dtm_tile_data(profile)
        geom_temp_doy = get_temperature_for_year(profile, dtm, lat_rows)
        del dtm, lat_rows
        landsat_data["geom_temp_doy"] = geom_temp_doy
        print(f"Time taken: {time.time() - time1} seconds")
        print("--------------------------------")

        print("Masking ...")
        time1 = time.time()
        valid_values_mask = landsat_data["modis_ndvi"] > 0
        _inplace_bitwise_and(valid_values_mask, landsat_data["mask"])
        _inplace_bitwise_and(
            valid_values_mask, np.isfinite(landsat_data["geom_temp_doy"]).all(axis=0)
        )

        n_valid_values = np.sum(valid_values_mask, axis=0)
        mask_valid_pixels = n_valid_values >= N_NEEDED_VALID_DATES

        landsat_data["mask_valid_pixels"] = mask_valid_pixels
        landsat_data["valid_values_mask"] = valid_values_mask
        # n_valid_pixels = inds_valid_pixels.size
        print(f"Number of valid pixels: {np.sum(mask_valid_pixels)}")
        print(f"Time taken: {time.time() - time1} seconds")
        print("================================")

    print(f"Total time taken for {len(tiles)}: {time.time() - time0} seconds")


def razno(landsat_data: dict[str, np.ndarray]) -> None:
    # %%
    import matplotlib.pyplot as plt

    red = landsat_data["red"]
    mask = landsat_data["mask"]
    ndvi = landsat_data["ndvi"]

    ind = 103
    plt.figure(figsize=(12, 6))
    plt.imshow(mask[ind, :].reshape((Y_SIZE, X_SIZE)))
    plt.title("MASK")
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.imshow(ndvi[ind, :].reshape((Y_SIZE, X_SIZE)), cmap="RdYlGn", vmin=-1, vmax=1)
    plt.title("NDVI")
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.imshow(red[ind, :].reshape((Y_SIZE, X_SIZE)), cmap="RdYlGn", vmin=0, vmax=1)
    plt.title("Red Band")
    plt.colorbar()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Red band - original")
    plt.imshow(red[ind, :].reshape((Y_SIZE, X_SIZE)), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Red band - masked")
    red_masked = np.copy(red[ind, :]).astype(np.float32)
    red_masked[~mask[ind, :]] = np.nan
    plt.imshow(red_masked.reshape((Y_SIZE, X_SIZE)), cmap="gray")
    plt.show()

    # %%


if __name__ == "__main__":
    test_get_all_data()
# %%
