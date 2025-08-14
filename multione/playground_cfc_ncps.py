#%%

import ncps
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
    global crs, transform, bounds, landsat_data, modis_data, covariate_data, covariate_names

    utils.ttprint(f"Loading tile {landsat_tile}")
    start = start0 = time.time()
    landsat_files = utils.get_landsat_filenames_gaia(landsat_tile, years)    
    landsat_data, crs, transform, bounds = utils.get_landsat_data(landsat_files, years)
    utils.ttprint(f"Landsat data loaded in {time.time() - start:.2f} seconds")
    start1 = time.time()
    modis_data = utils.get_modis_ndvi_data_rio(landsat_files, years, crs, bounds)
    utils.ttprint(f"MODIS NDVI data loaded in {time.time() - start1:.2f} seconds")  
    utils.ttprint(f"Landsat + modis: {time.time() - start:.2f} seconds")

    # masking
    print("Masking data ...")
    start = time.time()
    landsat_data = utils.mask_from_qa(landsat_data, len(years))
    utils.ttprint(f"Masked Landsat data from QA in {time.time() - start:.2f} seconds")  
    start1 = time.time()
    landsat_data = utils.mask_from_modis(landsat_data, modis_data, len(years))
    utils.ttprint(f"Masked Landsat data from MODIS in {time.time() - start1:.2f} seconds")
    print(f"Masked Landsat data in {time.time() - start:.2f} seconds")

    # get covariates
    print("Getting covariates ...")
    start = time.time()
    covariate_data, covariate_names = utils.get_dtm_covariates(landsat_files)
    print(f"Got covariates in {time.time() - start:.2f} seconds")

    stop = time.time()
    eta = stop - start0
    print(f"Total time for loading tile {landsat_tile}: {eta:.2f} seconds")

#   Landsat + modis: 320.55 seconds, Masked Landsat data in 134.58 seconds, Got covariates in 43.86 seconds
#   Total time for loading tile 055W_06S: 499.00 seconds

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
years = np.arange(2000, 2024)

get_all_data(landsat_tile, years)

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
    def __init__(self, years, sequence_length, landsat_data, modis_data, covariate_data, covariate_names, pixels, samples_per_pixel=100):
        self.years = years
        self.sequence_length = sequence_length
        self.landsat_data = landsat_data
        self.modis_data = modis_data
        self.covariate_data = covariate_data
        self.covariate_names = covariate_names        
        
        self.n_output_bands = 1
        self.n_features = 3     # za sada samo modis NDVi + landsat (isto kao i output) + doy

        nyears = len(years)
        self.ndates = nyears * utils.n_imag_per_year
        dates, _, doy = utils.get_dates_doy(years)

        self.dates = np.array(dates).astype('datetime64[D]')
        self.doy = np.array(doy)
        self.first_date = self.dates[0]
        self.last_date = self.dates[-1]

        if isinstance(pixels, int):
            # pixels = random number of pixels to sample
            n = 2
            while True:
                inds = np.random.choice(utils.n_pix, n*pixels, replace=False)   
                # Its enough to test only one band (first one) 
                valid_pixels = np.isfinite(self.landsat_data[:self.ndates, inds]).sum(axis=0) > samples_per_pixel # Check if we have enough pixels
                n_valid_pixels = valid_pixels.sum()
                if n_valid_pixels >= pixels:
                    inds = inds[valid_pixels][:pixels]
                    break
                else:
                    print(f"{n*pixels} random samples was not enough, found {n_valid_pixels}, lets try with {n*pixels*2}")
                    n *= 2
                    if n*pixels*3>utils.n_pix:
                        print(f"Not enough valid pixels in Landsat data for {pixels} samples. Found only {n_valid_pixels} valid pixels.")
                        inds = inds[valid_pixels]
                        break
        elif isinstance(pixels, (list, np.ndarray)):
            inds = np.array(pixels)
        else:
            raise ValueError(f"Invalid pixels argument: {pixels}")

        self.inds = inds

    def get_data_for_pixel(self, idx):
        # Get the data for a specific pixel
        return self.landsat_data[:self.ndates, self.inds[idx]], self.modis_data[:, self.inds[idx]]

    def __len__(self) -> int:
        return len(self.inds)

    def __getitem__(self, idx):
        ind = self.inds[idx]

        # Izlaz treba biti shape-a (B,L,C) ako je batch_first==True, inaće je (L,B,C)
        # Ako ne radimo batchave onda može i L, C
        # B - broj case-eva u batchu
        # C - broj featura
        # L - dužina sekvence
        # još bi trebalo vratiti i timespan shape-a (B, L) ili (L, B) ili samo L, to bi bio vremenski razmak od
        # prethodnog vremena
        # Batch bi zapravo mogao biti sve timeserije u jednom pixelu !!!

        # All bands should be included ...
        lsdata = self.landsat_data[:self.ndates, ind].reshape(-1,1) # shape = [ndates, nbands]
        msdata = self.modis_data[:, ind]    # alwajs only one band - ndvi

        # How many time series?
        valid_values = ~(np.isnan(lsdata[:,0]) | np.isnan(msdata[:]))
        nvv = valid_values.sum()
        nts = nvv - self.sequence_length  # number of time series in this pixel == number of batches
        if nts <= 0:
            raise ValueError(f"Not enough valid values for sequence length {self.sequence_length} in pixel {ind}")
        
        # TODO: skaliranje za svaki band !!! Možda staviti u __init__?
        lsdata = lsdata[valid_values,:]   
        msdata = msdata[valid_values]
        dates = self.dates[valid_values]  # valid dates for this pixel
        #dates = np.r_[vdates[0], vdates]  # add first date to the end for timespan calculation (only for forward case)
        vdoys = self.doy[valid_values]

        y = np.empty((nts, self.n_output_bands), dtype=np.float32)  # (B, C)
        timespans = np.empty((nts, self.sequence_length), dtype=np.float32)  # (L, 1)
        x = np.empty((nts, self.sequence_length, self.n_features), dtype=np.float32)  # (L, C)

        for i in range(nts):
            y[i] = lsdata[i+self.sequence_length, :].reshape(1, self.n_output_bands)
            timespans[i] = dates[i+1:i+1+self.sequence_length] - dates[i:i+self.sequence_length]
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
def test_dataset():
    #%%
    ds = TestDataset(years, 12, landsat_data, modis_data, covariate_data, covariate_names, pixels=100, samples_per_pixel=100)


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
        x, y = batch
        y_hat, _ = self.model.forward(x)
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
    ds = TestDataset(years, 12, landsat_data, modis_data, covariate_data, covariate_names, pixels=100, samples_per_pixel=100)
    #dataloader = data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    out_features = 1
    in_features = ds.n_features
    N = ds.sequence_length

    hparams={
        "lr":0.01, 
        "hidden_size": 8,
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


    trainer = pl.Trainer(
        logger=pl.loggers.CSVLogger("log"),
        max_epochs=100,
        gradient_clip_val=1,  # Clip gradient to stabilize training
        #gpus = 0
    )

    trainer.fit(learn, ds)

#%%
import matplotlib.pyplot as plt
idx = 0
lsdata, msdata = ds.landsat_data[:ds.ndates, ds.inds[idx]], ds.modis_data[:, ds.inds[idx]]
doy = ds.doy
dates = ds.dates

y_cnt = np.zeros_like(dates, dtype=np.int32)
y_sum = np.zeros_like(dates, dtype=np.float32)

valid_values = ~(np.isnan(lsdata[:,0]) | np.isnan(msdata[:]))
vdoys = ds.doy[valid_values]
vdates = ds.dates[valid_values]


# forward
for i in range(len(dates) - ds.sequence_length):





plt.figure(figsize=(20, 6))
plt.plot(dates, ld/2000, 'b+', label='Landsat Data')
plt.plot(dates, md/10000, 'g.',label='MODIS NDVI')
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
