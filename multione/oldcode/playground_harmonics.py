#%%
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression, Lasso

import utils, processing_utils
import time

landsat_tile = '055W_06S' # Brazil
years = range(2000, 2010) # type: ignore
#%% Load data
start = time.time()
utils.ttprint(f"Loading masked data from zarr for tile {landsat_tile} and years {years} ...")
arrays = utils.load_from_zarr_parallel(f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}.zarr')
landsat_data: NDArray[np.float32] = arrays['landsat_data']  # type: ignore
modis_data: NDArray[np.float32] = arrays['modis_data']  # type: ignore
years: NDArray[np.int32] = arrays['years']  # type: ignore 
utils.ttprint(f"Loaded data from zarr in {time.time() - start} seconds")
#%%
r, c = np.random.randint(0, utils.y_size), np.random.randint(0, utils.x_size)
ind = c + r * utils.x_size
utils.ttprint(f"Random pixel at row {r}, column {c}, index {ind}")
ts_modis = modis_data[:, ind]/10000
ts_landsat = landsat_data[:, ind]/10000
utils.ttprint(f"MODIS time series n: {np.isfinite(ts_modis).sum()}")
utils.ttprint(f"Landsat time series n: {np.isfinite(ts_landsat).sum()}")
utils.show_timeseries(landsat_data, modis_data, years, r, c, bands_to_show=[8]) #type: ignore
#plt.figure(figsize=(10, 6))
#plt.plot(ts_modis, 'x-', label='MODIS NDVI', markersize=3)
#plt.plot(ts_landsat, 'o-', label='Landsat NDVI', markersize=3)
#plt.legend()
#plt.show()
# %%
max_order = 5 # Maximum order of harmonic function
ns = 1 # Number of seasons
T = 365 / ns # Length of one season in days
dates_list = []
for y in years:
    # y= 2001
    dates = [datetime(y,1,8)+timedelta(days=16*i) for i in range(utils.n_imag_per_year)]
    dates_list.extend(dates)
t = np.array([(d - datetime(years[0],1,1)).days for d in dates_list])
#doy = np.array([(d - datetime(d.year, 1, 1)).days + 1 for d in dates_list])
pi_t_T = 2 * np.pi * t / T

fig = plt.figure(figsize=(10, 6))
plt.plot(t, ts_modis, 'x', label='MODIS', markersize=3)

predictors = t.reshape(-1, 1)  # Reshape t to be a column vector
for order in range(1, max_order + 1):
    utils.ttprint(f"Fitting harmonic model with order {order} ...")
    predictors = np.c_[predictors, np.cos(order * pi_t_T), np.sin(order * pi_t_T)]
    #reg = LinearRegression().fit(predictors, ts_modis)
    reg = Lasso(alpha=0.001, selection='random').fit(predictors, ts_modis)  # Using Lasso regression for regularization
    fitted = reg.predict(predictors)
    plt.plot(t, fitted, label=f'Order {order}, r^2= {reg.score(predictors, ts_modis):.3f}')

plt.xlabel('Days since 2001-01-01')
plt.ylabel('MODIS NDVI')
plt.legend()
plt.show()
# %% HANTS

prd, outliers = HANTS(
    ni=len(ts_modis), 
    nb=365, 
    nf=5, 
    y=ts_modis, 
    ts=t, 
    HiLo = 'Lo', #None, #'Hi', 
    low=0.1, high=0.9, 
    fet=0.05, 
    dod=0, 
    delta=0.1, 
    fill_val=np.nan)

fig = plt.figure(figsize=(10, 6))
plt.plot(t, ts_modis, '.', label='MODIS', markersize=3)
plt.plot(t, prd, 'o-', label='HANTS', markersize=3)
outliers_mask = outliers[0] != 0
plt.plot(t[outliers_mask], ts_modis[outliers_mask], 'rx', label='Outliers', markersize=5)
plt.xlabel('Days since 2001-01-01')
plt.ylabel('MODIS NDVI')
plt.legend()
# %%
