#%%
import gc
import json
import time
from pathlib import Path

import data_utils as dutils
import production_utils as putils
from settings import DATES_TO_PREDICT, MASTER_LOGGER_FILE, MODEL_OPTIMIZATION, N_NEEDED_VALID_DATES, PREDICTIONS_FOLDER, SETTINGS_DICT, FILE_ENDING_OUT, MODEL_NAME

# putils.init()
fld = Path('/mnt/nibble/gen_cog/arcov2/averages')
log_file = fld/'master_average.log'
# %%
def average_monthly_production(tile: str) -> None:
    # tile = '055W_06S'
    log = dutils.ProductionLogger(log_file, silent=False)
    #data = dutils.load_tile_data(tile, log)
    # Need to make new function to load only needed data for averages (all channels, no ndvi, no timeless features)



# tiles=['055W_06S','090W_49N','015W_43N']