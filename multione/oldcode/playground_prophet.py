#%%
import importlib
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import utils, processing_utils
import time
import matplotlib.pyplot as plt

landsat_tile = '055W_06S' # Brazil
years = np.arange(2000, 2010) # type: ignore
#%% reimport utils

utils = importlib.reload(utils)

#%% Load data
start = time.time()
utils.ttprint(f"Loading masked data from zarr for tile {landsat_tile} and years {years} ...")
arrays = utils.load_from_zarr_parallel(f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}.zarr')
landsat_data: NDArray[np.float32] = arrays['landsat_data']  # type: ignore
modis_data: NDArray[np.float32] = arrays['modis_data']  # type: ignore
years: NDArray[np.int32] = arrays['years']  # type: ignore 
utils.ttprint(f"Loaded data from zarr in {time.time() - start} seconds")
#%% Get dtm derivatives
landsat_files = utils.get_landsat_filenames_local(landsat_tile, years, '/mnt/nibble/gen_cog/arcov2')
#%%
dtm_derivatives, dtm_derivatives_names = utils.get_dtm_derivatives(landsat_files)
dtm, lat_rows = utils.get_dtm_data(landsat_files)
temp_min, temp_max = processing_utils.temperature_min_max(dtm, lat_rows)

covariate_data = np.concatenate((dtm_derivatives, dtm.reshape(1,-1), temp_min.reshape(1,-1), temp_max.reshape(1,-1)), axis=0)
covariate_names = dtm_derivatives_names + ['dtm', 'temp_min', 'temp_max']

del dtm_derivatives, dtm, lat_rows, temp_min, temp_max
#%% Show DTM derivatives