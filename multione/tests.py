#%% Loading zarr data parallelly
from numpy.typing import NDArray, ArrayLike
import numpy as np
import utils
import time

landsat_tile = '055W_06S'
years = range(2000, 2010) # type: ignore

start = time.time()
utils.ttprint(f"Loading masked data from zarr for tile {landsat_tile} and years {years} ...")
arrays = utils.load_from_zarr_parallel(f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}.zarr')
landsat_data: NDArray[np.float32] = arrays['landsat_data']  # type: ignore
modis_data: NDArray[np.float32] = arrays['modis_data']  # type: ignore
years: NDArray[np.int32] = arrays['years']  # type: ignore 
utils.ttprint(f"Loaded data from zarr in {time.time() - start} seconds")

# n_threads: <200 seconds
# 2*n_threads: 172 seconds
#%%