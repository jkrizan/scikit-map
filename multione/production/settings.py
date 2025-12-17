from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import multiprocessing as mp

import rasterio
import rasterio.enums

# ###############################################
# Model
MODEL_SUBFOLDER = 'model'
MODEL_NAME = 'v0_xs_2_8_8' #'xs40'

MODEL_OUTPUT_SIZE = 1  # Number of output indices
MODEL_INPUT_SIZE = MODEL_OUTPUT_SIZE + 2  # Number of input bands + 2 time features
MODEL_TIMELESS_SIZE = 3  # Number of timeless input features
MODEL_SEQUENCE_LENGTH = 12  # Number of time steps in the input sequence
MODEL_OPTIMIZATION = "OPENVINO"  # Options: NONE, OPENVINO
MODEL_PREDICTION_MIN = 0.0
MODEL_PREDICTION_MAX = 1.0

################################################
# Production settings
PRODUCTION_FOLDER = Path('/mnt/nibble/gen_cog/arcov2/production')
PRODUCTION_MODEL_FOLDER = PRODUCTION_FOLDER / MODEL_NAME
DTM_FILE = '/global/dtm/v3/filtered.dtm_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20241230.tif'
DATASET_ZARR = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
TILES_FILE = PRODUCTION_FOLDER / 'tiles.txt'
TMP_FOLDER = PRODUCTION_FOLDER / 'tmp'
LOG_FOLDER = PRODUCTION_MODEL_FOLDER / 'logs'
PREDICTIONS_FOLDER = PRODUCTION_MODEL_FOLDER / f'predictions'
MASTER_LOGGER_FILE = PRODUCTION_MODEL_FOLDER / 'production_master.log'
LOGGER_SILENT = False

N_THREADS = mp.cpu_count()
N_THREADS_INFERENCE = N_THREADS // 2    # hyperthreading is not working !!!
PRODUCTION_BATCH_SIZE = N_THREADS_INFERENCE * 300
DEVICE = 'CPU'  # 'CPU' or 'CUDA'

YEARS = list(range(2000, 2024))
DATES_TO_PREDICT = [datetime(year, month, 15) for year in YEARS for month in range(1, 13)]
IMG_NODATA = 65000
IMG_DTYPE = np.uint16
IMG_SCALE_FACTOR = 40000
N_NEEDED_VALID_DATES = 14    # Minimum is 12 valid dates per pixel 
FILTER_DIFF_TH, FILTER_COUNT_TH = (2500 / 10000.0, 2*len(YEARS))
DATA_TYPE = np.float32 #np.float16
OUTPUT_PROFILE = dict(
    driver='GTiff',
    dtype='uint16',
    count=1,
    nodata=IMG_NODATA,
    width=4004,
    height=4004,
    crs='EPSG:4326',
    compress='deflate',
    predictor=2,
    tiled=True,
    blockxsize=1024,
    blockysize=1024,
)
MODIS_RESAMPLING_STRATEGY = rasterio.enums.Resampling.cubic_spline
RND_GEN = np.random.default_rng(42)

# ##############################################
# Constants
X_SIZE = Y_SIZE = 4004  # GLAD and MODIS images are 4004x4004 pixels
N_PIXELS = X_SIZE * Y_SIZE
N_IMAGES_PER_YEAR = 23
LANDSAT_FILE_ENDING = '_go_epsg.4326_v20240521.tif'
LANDSAT_SCALE_FACTOR = DATA_TYPE(1/40000)
LANDSAT_NODATA = 0
MODIS_SCALE_FACTOR = DATA_TYPE(1/10000)
MODIS_NODATA = -3000
FILE_ENDING_OUT = '_go_epsg.4326_v2'

# Gaia S3 parameters
GAIA_ADDRS = [f'http://192.168.49.{gaia_ip}:8333' for gaia_ip in range(30, 47)]
GAIA_S3_PARAMS = {
        's3_addresses':GAIA_ADDRS,
        's3_access_key':'iwum9G1fEQ920lYV4ol9',
        's3_secret_key':'GMBME3Wsm8S7mBXw3U4CNWurkzWMqGZ0n2rXHggS0',
        's3_prefix':'tmp-landsat-arco-v2',
    }

DTM_FILE = GAIA_ADDRS[RND_GEN.integers(0, len(GAIA_ADDRS))] + DTM_FILE

BANDS_PREFIX = ['red_glad',
                'nir_glad',
                'blue_glad',
                'green_glad',
                'swir1_glad',
                'swir2_glad',
                'thermal_glad',
                'qa_mask']

BANDS_DICT = {'red': 'red_glad',
              'nir': 'nir_glad',
              'blue': 'blue_glad',
              'green': 'green_glad',
              'swir1': 'swir1_glad',
              'swir2': 'swir2_glad',
              'thermal': 'thermal_glad',
              'qa': 'qa_mask'}

# Landsat time-series parameters
DOY_START = ['0101', '0117', '0202', '0218', '0305', '0321', '0406', '0422', '0508', '0524', '0609',
             '0625', '0711', '0727', '0812', '0828', '0913', '0929', '1015', '1031', '1116', '1202', '1218']

DOY_END = ['0116', '0201', '0217', '0304', '0320', '0405', '0421', '0507', '0523', '0608', '0624',
           '0710', '0726', '0811', '0827', '0912', '0928', '1014', '1030', '1115', '1201', '1217', '1231']

MONTH_START = ['0101', '0201', '0301', '0401', '0501', '0601', '0701', '0801', '0901', '1001', '1101', '1201']
MONTH_END = ['0131', '0228', '0331', '0430', '0531', '0630', '0731', '0831', '0930', '1031', '1130', '1231']

SETTINGS_DICT_KEYS = ['MODEL_NAME', 'MODEL_OPTIMIZATION', 'N_THREADS', 'N_THREADS_INFERENCE',
                       'PRODUCTION_BATCH_SIZE', 'N_NEEDED_VALID_DATES', 'FILTER_DIFF_TH', 'FILTER_COUNT_TH']


######################################

FIRST_IMAGE_DATE = datetime(YEARS[0], 1, 8)
IMAGE_DATES = [datetime(y,1,8) + timedelta(days=16 * i) for y in YEARS for i in range(N_IMAGES_PER_YEAR)]
DAYS_FROM_START = [(d - datetime(YEARS[0], 1, 1)).days for d in IMAGE_DATES]

SETTINGS_DICT = {k: globals()[k] for k in SETTINGS_DICT_KEYS}