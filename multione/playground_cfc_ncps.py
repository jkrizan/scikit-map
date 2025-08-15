#%%

from os import name
import ncps
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import gc

import utils, processing_utils
import time
import matplotlib.pyplot as plt
from pathlib import Path

#%%
import importlib
utils = importlib.reload(utils)

#%%
# Let's try to make some global model from ex. 10M samples
# So we need some number of random samples and some number of random points in samples
# 1 sample is one point in time, and there is 24 years x 23 images per year = 552 points in time
# There is 18667 tiles 
# 10M/18667 = 535 samples per tile, 535/552 = 0.97 - so its 1 sample per point in time
def get_all_data(landsat_tile: str, years: NDArray[np.int32]):
    # landsat_tile = '055W_06S'
    # years = np.arange(2000, 2024)
    global crs, transform, bounds, landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy

    utils.ttprint(f"Loading tile {landsat_tile}")
    start = start0 = time.time()
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
    landsat_data = utils.landsat_data_trim_scale(landsat_data, max_ind, 10000)    # scale all bands with 10000    #TODO: it can be parallized !!! (numba)
    utils.ttprint(f"Landsat data scaled and trimmed in {time.time() - start:.2f} seconds")
    
    # get covariates
    utils.ttprint("Getting covariates ...")
    start = time.time()
    covariate_data, covariate_names = utils.get_dtm_covariates(landsat_files)
    
    utils.ttprint(f"Covariates loaded in {time.time() - start:.2f} seconds")

    utils.ttprint("Getting geom_temp_doy ...")
    start = time.time()
    dtm_ind = covariate_names.index('dtm')
    geom_temp_doy = utils.get_temperature_for_year(landsat_files, covariate_data[dtm_ind])
    utils.ttprint(f"Got geom_temp_doy in {time.time() - start:.2f} seconds")

    stop = time.time()
    eta = stop - start0
    utils.ttprint(f"Total time for loading tile {landsat_tile}: {eta:.2f} seconds")

#   Landsat + modis: 320.55 seconds, Masked Landsat data in 134.58 seconds, Got covariates in 43.86 seconds
#   Total time for loading tile 055W_06S: 499.00 seconds

# Landsat data loaded in 25.70 seconds, MODIS NDVI data loaded in 167.20 second, Landsat + modis: 192.90 seconds
# Masked Landsat data from QA in 102.31 seconds, Masked Landsat data from MODIS in 44.99 seconds, Masked Landsat data in 147.30 seconds
# Got covariates in 60.83 seconds, .... Total time for loading tile 055W_06S: 401.04 seconds

#%%
def save_tile_data():
    import zarr
    start = time.time()
    root = zarr.group(f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}_all.zarr', overwrite=True)
    root.create_array('landsat_data', data=landsat_data)
    root.create_array('modis_data', data=modis_data)
    root.create_array('years', data=np.array(years))
    root.create_array('covariate_data', data=covariate_data)
    root.create_array('covariate_names', data=np.array(covariate_names))
    utils.ttprint(f"Saved masked Landsat data to zarr in {time.time() - start} seconds")

def load_tile_data():
#%%    
    landsat_tile = '055W_06S'
    start = time.time()
    utils.ttprint(f"Loading masked data from zarr for tile {landsat_tile}  ...")
    arrays = utils.load_from_zarr_parallel(f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}_all.zarr')
    landsat_data: NDArray[np.float32] = arrays['landsat_data']  # type: ignore
    modis_data: NDArray[np.float32] = arrays['modis_data']  # type: ignore
    years: NDArray[np.int32] = arrays['years']  # type: ignore
    covariate_data: NDArray[np.float32] = arrays['covariate_data']  # type: ignore
    covariate_names: NDArray[np.str_] = arrays['covariate_names']  # type: ignore
    utils.ttprint(f"Loaded data from zarr in {time.time() - start} seconds")
    # Loaded data from zarr in 166.3748698234558 seconds
    # Loaded data from zarr in 490.163366317749 seconds

    #landsat_files = utils.get_landsat_filenames_local(landsat_tile, years, '/mnt/nibble/gen_cog/arcov2')
    #covariate_data, covariate_names = utils.get_dtm_covariates(landsat_files)


#%%
landsat_tile = '055W_06S'
years = np.arange(2000, 2025)

get_all_data(landsat_tile, years)

# memory:
# landsat_data = 300GB, 
# landsat_data.shape[0]/(utils.n_imag_per_year*len(years)) == 9, 7 bands, QA, NDVI
# modis_data = 33GB
# covariate_data = 50GB
# geom_temp_doy = 10GB
# total = 460GB 
#%% find memory clogs
from operator import itemgetter
from pympler import tracker
from pympler import asizeof

mem = tracker.SummaryTracker()
print(sorted(mem.create_summary(), reverse=True, key=itemgetter(2))[:10])

objs = [(name, asizeof.asizeof(obj)/1024**3, id(obj)) for name, obj in globals().items() if isinstance(obj, (np.ndarray, list, dict))]
print('\n'.join(map(str,sorted(objs, reverse=True, key=itemgetter(1)))))

import threading
print(f"Threading: {threading.active_count()} threads active")

for thread in threading.enumerate():
    print(f" - {thread.name}")
#%%
import importlib

import ncps
CfC = importlib.reload(ncps).torch.CfC

importlib.invalidate_caches()
#%%
import numpy as np
import torch.nn as nn
from ncps.wirings import AutoNCP
from ncps.torch import LTC, CfC
import pytorch_lightning as pl
import torch
import torch.utils.data as data

class TestDataset(data.Dataset):
    def __init__(self, years, sequence_length, landsat_data, modis_data, covariate_data, covariate_names, train_pixels, test_pixels=0.2, samples_per_pixel=100):
        self.years = years
        self.sequence_length = sequence_length
        self.landsat_data = landsat_data
        self.modis_data = modis_data
        self.covariate_data = covariate_data
        self.covariate_names = covariate_names        
        
        self.n_output_bands = 7
        self.n_features = 2 + self.n_output_bands  # modis_ndvi + geom_temp + landsat_bands

        nyears = len(years)
        self.ndates = nyears * utils.n_imag_per_year
        dates, _, doy = utils.get_dates_doy(years)

        self.dates = np.array(dates).astype('datetime64[D]')
        self.doy = np.array(doy)
        self.first_date = self.dates[0]
        self.last_date = self.dates[-1]

        self.valid_values_mask = np.isfinite(self.landsat_data[:self.ndates])
        self.n_valid_values = np.isfinite(self.landsat_data[:self.ndates]).sum(axis=0)   
        # 16s
        self.inds_valid_pixels = np.where(self.n_valid_values > samples_per_pixel)[0]  # indices of valid pixels with enough samples
        self.n_valid_pixels = len(self.inds_valid_pixels)


        if isinstance(train_pixels, int):
            # pixels = random number of pixels to sample
            self.train_pixels = train_pixels
            self.test_pixels = int(test_pixels * self.train_pixels) if test_pixels < 1 else test_pixels
            if (self.n_valid_pixels < train_pixels + self.test_pixels):
                raise ValueError(f"Not enough valid pixels in Landsat data for {train_pixels} samples. Found only {self.n_valid_pixels} valid pixels.")

            all_inds = np.random.choice(self.inds_valid_pixels, size=self.train_pixels + self.test_pixels, replace=False)  # type: ignore #  random pixels with enough samples
            self.train_inds = all_inds[:self.train_pixels]
            self.test_inds = all_inds[self.train_pixels:self.train_pixels+self.test_pixels]

        elif isinstance(train_pixels, (list, np.ndarray)):
            self.train_inds = np.array(train_pixels)
        else:
            raise ValueError(f"Invalid pixels argument: {train_pixels}")

    def get_test_dataset(self):
        return TestDataset(self.years, self.sequence_length, self.landsat_data, self.modis_data, self.covariate_data, self.covariate_names, train_pixels=self.test_inds, test_pixels=0)

    #def get_data_for_pixel(self, idx):
        # Get the data for a specific pixel
    #    return self.landsat_data[:self.ndates, self.inds[idx]], self.modis_data[:, self.inds[idx]]

    def __len__(self) -> int:
        return len(self.train_inds)

    def __getitem__(self, idx):
        ind = self.train_inds[idx]

        # Izlaz treba biti shape-a (B,L,C) ako je batch_first==True, inaće je (L,B,C)
        # Ako ne radimo batchave onda može i L, C
        # B - broj case-eva u batchu
        # C - broj featura
        # L - dužina sekvence
        # još bi trebalo vratiti i timespan shape-a (B, L) ili (L, B) ili samo L, to bi bio vremenski razmak od
        # prethodnog vremena
        # Batch bi zapravo mogao biti sve timeserije u jednom pixelu !!!

        # TODO: All bands should be included ...
        lsdata = self.landsat_data[:self.ndates, ind].reshape(-1,1) # shape = [ndates, nbands]
        msdata = self.modis_data[:, ind]    # always only one band - ndvi

        # How many time series?
        valid_values = ~(np.isnan(lsdata[:,0]) | np.isnan(msdata[:]))        
        nvv = valid_values.sum()
        nts = nvv - self.sequence_length  # number of time series in this pixel == number of batches
        if nts <= 0:
            raise ValueError(f"Not enough valid values for sequence length {self.sequence_length} in pixel {ind}")
        
        # TODO: skaliranje za svaki band !!! Možda staviti u __init__?
        lsdata = lsdata[valid_values,:] / 10000
        msdata = msdata[valid_values] / 10000
        dates = self.dates[valid_values]  # valid dates for this pixel
        #dates = np.r_[vdates[0], vdates]  # add first date to the end for timespan calculation (only for forward case)
        vdoys = self.doy[valid_values]/366

        y = np.empty((nts, self.n_output_bands), dtype=np.float32)  # (B, C)
        timespans = np.empty((nts, self.sequence_length), dtype=np.float32)  # (L, 1)
        x = np.empty((nts, self.sequence_length, self.n_features), dtype=np.float32)  # (L, C)

        for i in range(nts):
            timespans[i] = (dates[i+1:i+1+self.sequence_length] - dates[i:i+self.sequence_length])/366
            # TODO: exclude timeseries with any timspan>1
            # TODO: implement scaling for each band
            y[i] = lsdata[i+self.sequence_length, :].reshape(1, self.n_output_bands)
            #x[i] = np.concatenate((self.covariate_data[:,ind], self.modis_data[:, ind].reshape(-1, 1)), axis=0)
            x[i] = np.concatenate((vdoys[i:i+self.sequence_length].reshape(-1, 1),
                                   msdata[i:i+self.sequence_length].reshape(-1, 1),                                    
                                   lsdata[i:i+self.sequence_length]), axis=1)

        return idx, torch.tensor(x), torch.tensor(y), torch.tensor(timespans).view(nts, self.sequence_length,1)

        # # Ovo treba pak raspakirati ... za sad vraćam samo 1. band
        # y = self.landsat_data[:self.ndates, ind]   # (230,)

        # covdata = self.covariate_data[:, ind]  # (17,)
        # modisndvi = self.modis_data[:, ind].reshape(-1, 1) # (230, 1)

        # x = np.concatenate((self.covariate_data[:,ind], self.modis_data[:, ind].reshape(-1, 1)), axis=0)


#%%
# LightningModule for training a RNNSequence module
class SequenceLearner(pl.LightningModule):
    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
        #self.lr = lr
        self._hparams = hparams

    def training_step(self, batch, batch_idx):
        # batch = ds[0]
        ix, x, y, t = batch
        y_hat, (hc,cx) = self.model.forward(x, timespans = t)
        # y_hat, (hx,cx) = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        ix, x, y, t = batch
        y_hat, _ = self.model.forward(x, timespans = t)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self._hparams['lr'])
    
#%%  
def train():
#%%
    utils.ttprint("Constructing dataset ...")
    ds_train = TestDataset(years, 12, landsat_data, modis_data, covariate_data, covariate_names, train_pixels=2000, test_pixels=0.2, samples_per_pixel=100)
    ds_test = ds_train.get_test_dataset()  # use the same dataset for testing
    #dataloader = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    out_features = 1
    in_features = ds_train.n_features
    N = ds_train.sequence_length

    hparams={
        "lr":0.01, 
        "hidden_size": 16,
        "backbone_units": 8,
        "backbone_layers": 2,
        "backbone_dropout": 0.1
    }

    #wiring = AutoNCP(16, out_features)  # 16 units, 1 motor neuron
    #ltc_model = LTC(in_features, wiring, batch_first=True)
    model = CfC(
        input_size = in_features,
        units = hparams["hidden_size"],
        proj_size = out_features,
        return_sequences=False,
        batch_first=True,
        mixed_memory=True,
        backbone_units=hparams["backbone_units"],
        backbone_layers=hparams["backbone_layers"],
        backbone_dropout=hparams["backbone_dropout"]
    )
    learn = SequenceLearner(model, hparams=hparams)

    utils.ttprint("Training model ...")
    trainer = pl.Trainer(
        logger=pl.loggers.CSVLogger("log"),
        max_epochs=10,
        gradient_clip_val=1,  # Clip gradient to stabilize training
        #gpus = 0
    )

    trainer.fit(learn, train_dataloaders=ds_train, val_dataloaders=ds_test)
    utils.ttprint("Training finished.")
    # 10 epochs - 5min

#%%
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error

idx = 2
ind = ds_train.test_inds[idx]
pixel_mask = ds_train.valid_values_mask[:,ind]

lsdata = ds_train.landsat_data[:ds_train.ndates, ind]/10000
msdata = ds_train.modis_data[:, ind]/10000
doy = ds_train.doy/366
dates = ds_train.dates
pixel_inds = np.where(pixel_mask)[0]

y_cnt = np.zeros_like(dates, dtype=np.int32)
y_sum = np.zeros_like(dates, dtype=np.float32)

nts = pixel_mask.sum() - ds_train.sequence_length

# forward pass
for i in range(nts+1):  # +1 if true values don't go to the end ...
    # i=1
    i_pixel_inds = pixel_inds[i:i+ds_train.sequence_length]
    #if pixel_inds[i+ds_train.sequence_length]>i_pixel_inds[-1]+1:
    x = np.concatenate((doy[i_pixel_inds].reshape(-1, 1),
                        msdata[i_pixel_inds].reshape(-1, 1),
                        lsdata[i_pixel_inds].reshape(-1,1)), axis=1).astype(np.float32)
    i_dates = dates[i_pixel_inds]
    timespan = np.r_[i_dates[1:] - i_dates[:-1],0].astype(np.float32)
    if i==nts:
        ind_last = ds_train.ndates
    else:
        ind_last= pixel_inds[i+ds_train.sequence_length]+1
    for j in range(i_pixel_inds[-1] + 1, ind_last):
        # j = i_pixel_inds[-1]+1
        timespan[-1] += 16
        # return idx, torch.tensor(x), torch.tensor(y), torch.tensor(timespans).view(nts, self.sequence_length,1)
        # y_hat, (hc,cx) = self.model.forward(x, timespans = t)
        y_hat,_ = learn.model.forward(torch.tensor(x).unsqueeze(0), None, torch.tensor(timespan/366).unsqueeze(0)) #.view(ds_train.sequence_length, 1)))
        y_cnt[j] += 1
        y_sum[j] += y_hat


y_valid = y_cnt > 0
y_mask = y_valid & pixel_mask

y_prd = y_sum[y_mask]/y_cnt[y_mask]
y_obs = lsdata[y_mask]
loss = nn.MSELoss()(torch.tensor(y_obs),torch.tensor(y_prd))
mse = mean_squared_error(y_obs, y_prd)
r2 = r2_score(y_obs, y_prd) 
mae = mean_absolute_error(y_obs, y_prd)

print(f"Loss: {loss:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")

y_prd = y_sum[y_valid]/y_cnt[y_valid]

plt.figure(figsize=(20, 6))
plt.plot(dates, lsdata, 'b+', label='Landsat Data')
plt.plot(dates, msdata/4, 'g.',label='MODIS NDVI')
plt.plot(dates[y_valid], y_prd, 'r-', label='Predicted')
plt.legend()

#%%

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("white")
plt.figure(figsize=(6, 4))
legend_handles = wiring.draw_graph(draw_labels=True, neuron_colors={"command": "tab:cyan"})
plt.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(1, 1))
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()
#%%
