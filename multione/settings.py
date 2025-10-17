
from typing import Tuple, List
import os
from pathlib import Path
import pandas
import numpy as np
import socket


n_threads = 96

os.environ['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
os.environ['USE_PYGEOS'] = '0'
os.environ['PROJ_LIB'] = '/opt/conda/share/proj/'
os.environ['NUMEXPR_MAX_THREADS'] = f'{n_threads}'
os.environ['NUMEXPR_NUM_THREADS'] = f'{n_threads}'
os.environ['OMP_THREAD_LIMIT'] = f'{n_threads}'
os.environ["OMP_NUM_THREADS"] = f'{n_threads}'
os.environ["OPENBLAS_NUM_THREADS"] = f'{n_threads}' # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = f'{n_threads}' # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = f'{n_threads}'

TMP_DIR = '/mnt/silva/tmp'

# Gaia S3 parameters
gaia_addrs = [f'http://192.168.49.{gaia_ip}:8333' for gaia_ip in range(30, 47)]
gaia_s3_params = {
        's3_addresses':gaia_addrs,
        's3_access_key':'iwum9G1fEQ920lYV4ol9',
        's3_secret_key':'GMBME3Wsm8S7mBXw3U4CNWurkzWMqGZ0n2rXHggS0',
        's3_prefix':'tmp-landsat-arco-v2',
    }

gaia_addrs = [f'http://192.168.49.{gaia_ip}:8333' for gaia_ip in range(30, 47)]

# S3 parameters for MinIO Client (mc)
s3_params = {
    's3_addresses':gaia_addrs,
    's3_access_key':'iwum9G1fEQ920lYV4ol9',
    's3_secret_key':'GMBME3Wsm8S7mBXw3U4CNWurkzWMqGZ0n2rXHggS0',
    's3_prefix':'tmp-landsat-arco-v2',
}

# Function to set up S3 aliases using MinIO Client (mc)
# This function takes access key, secret key, and a list of Gaia addresses,
def s3_setup(access_key, secret_key, gaia_addrs, sudo=True) -> List[str]:
    import subprocess

    s3_aliases = []
    s3_aliases = [f'g{i+1}' for i, _ in enumerate(gaia_addrs)]
    commands = [
        f'{ "sudo" if sudo else "" } mc alias set  g{i+1} {addr} {access_key} {secret_key} --api S3v4'
        for i, addr in enumerate(gaia_addrs)
    ]
    for cmd in commands:
        subprocess.run(cmd, shell=True, capture_output=False, text=True, check=True)
    return s3_aliases

if not socket.gethostname() in('ceres','hydra'):
    s3_aliases = s3_setup(s3_params['s3_access_key'],
             s3_params['s3_secret_key'],
             s3_params['s3_addresses'],
             sudo=False if socket.gethostname() in ('mo-arcov2-compute') else True)
else:
    print('No S3 setup here !')
    s3_aliases = []

## LULC parameters
if not socket.gethostname() in ('ceres', 'hydra','mo-arcov2-compute'):
    lulc_base_path = Path('http://192.168.1.30:8333/global/lc/')
    lulc_filenames = {year: lulc_base_path/f'lc_glad.glcluc_c_30m_s_{year}0101_{year}1231_go_epsg.4326_v2.tif' for year in range(2015, 2024)}
    lulc_default_year = 2015  # Default year for LULC data
    for year in range(2000,2015):
        lulc_filenames[year] = lulc_base_path/f'lc_glad.glcluc_c_30m_s_{lulc_default_year}0101_{lulc_default_year}1231_go_epsg.4326_v2.tif'
    lulc_legend_filename = '/mnt/nibble/gen_cog/arcov2/legend_glcluc.xlsx'
    df = pandas.read_excel(lulc_legend_filename)
    class_names, ind, indinv =  np.unique(df.class1, return_index=True, return_inverse=True)
    # TODO: završiti legendu, dodati i druge klase
else:
    lulc_base_path = lulc_filenames = lulc_default_year = lulc_legend_filename = None
    print('LULC setup not done !')


## DEM parameters
dtm_adresses = gaia_addrs
dtm_vars = dict(
    dtmv1 = '/global/dtm/filtered.dtm_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
    dtmv3 = '/global/dtm/v3/filtered.dtm_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20241230.tif',
    rf = '/global/edtm/legendtm_rf_30m_m_s_20000101_20231231_go_epsg.4326_v20250130.tif',
    popen = '/global/dtm/pos.openness_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
    nopen = '/global/dtm/neg.openness_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20240528.tif',
    slope = '/global/dtm/slope_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20240528.tif',    
    dfme = '/global/dtm/v3/dfme_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif',
    geomorphon = '/global/dtm/v3/geomorphon_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif',
    # hs = '/global/dtm/v3/hillshade_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif',    
    lsf = '/global/dtm/v3/ls.factor_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif',
    maxic = '/global/dtm/v3/maxic_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif',
    minic = '/global/dtm/v3/minic_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif',
    procurv = '/global/dtm/v3/pro.curv_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif',
    ringcurv = '/global/dtm/v3/ring.curv_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif',
    shpindx = '/global/dtm/v3/shpindx_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif',
    tancurv = '/global/dtm/v3/tan.curv_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif',
    twi = '/global/dtm/v3/twi_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif',
)


# GDAL options for reading and writing
gdal_opts = {
 'GDAL_HTTP_VERSION': '1.0',
 'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tif',
}

no_data_out = 65000


gdal_co = ['TILED=YES', 'BIGTIFF=YES', 'COMPRESS=DEFLATE', 'BLOCKXSIZE=1024', 'BLOCKYSIZE=1024']

bands_prefix = ['red_glad',
                'nir_glad',
                'blue_glad',
                'green_glad',
                'swir1_glad',
                'swir2_glad',
                'thermal_glad',
                'qa_mask']
n_spect_bands = len(bands_prefix) - 1  # Exclude 'qa_mask' band
bands_scales_real = [10000, 40000, 10000, 10000, 40000, 40000, 30000, 1, 10000]

bands_prefix_out = ['red_glad',
                    'nir_glad',
                    'blue_glad',
                    'green_glad',
                    'swir1_glad',
                    'swir2_glad',
                    'thermal_glad']

file_ending_out = '_go_epsg.4326_v7'

# Landsat time-series parameters
doy_start = ['0101', '0117', '0202', '0218', '0305', '0321', '0406', '0422', '0508', '0524', '0609',
             '0625', '0711', '0727', '0812', '0828', '0913', '0929', '1015', '1031', '1116', '1202', '1218']

doy_end = ['0116', '0201', '0217', '0304', '0320', '0405', '0421', '0507', '0523', '0608', '0624',
           '0710', '0726', '0811', '0827', '0912', '0928', '1014', '1030', '1115', '1201', '1217', '1231']

month_start = ['0101', '0201', '0301', '0401', '0501', '0601', '0701', '0801', '0901', '1001', '1101', '1201']
month_end = ['0131', '0228', '0331', '0430', '0531', '0630', '0731', '0831', '0930', '1031', '1130', '1231']

# SWA time-series reconstruction parameters
att_env, att_seas, future_scaling = (20.0, 40.0, 0.1)

# MODIS NDVI filtering parameters
# diff_th, count_th = (3000, int(0.3*n_s))
# diff_th, count_th = (1000, 12*n_years)
#diff_th, count_th = (2500, 2*n_years)

resampling_strategy = "GRA_Bilinear"

x_off = y_off = 0  # GLAD and MODIS images are aligned to the top-left corner
x_size = y_size = 4004  # GLAD and MODIS images are 4004x4004 pixels
n_pix = x_size * y_size
n_imag_per_year = 23
n_imag_per_year_agg = 12
no_data = 0

landsat_file_ending = '_go_epsg.4326_v20240521.tif'

# Masking parameters
mask_band_scaling = 1/4e4
mask_result_scaling = 1e4
mask_result_offset = 0.

# Impainting stripes parameters
fft_th = 1.5
gap_stripes_th, gap_general_th = 0.35, 0.05
inpaint_chunk_size = 128
inpaint_radius = 3
inpaint_padding = 15

# MODIS NDVI filtering parameters
# diff_th, count_th = (3000, int(0.3*n_s))
# diff_th, count_th = (1000, 12*n_years)
#filter_diff_th, filter_count_th = (2500, 2*n_years)
def filter_params(n_years: int ) -> Tuple[int, int]:
    """
    Returns the filtering parameters based on the number of years.
    """
    filter_diff_th, filter_count_th = (2500, 2*n_years)
    return filter_diff_th, filter_count_th