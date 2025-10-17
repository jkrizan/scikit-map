#%%

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

#%%
def tile_load_stats():
    # Find all tiles in the bucket
    import subprocess, re

    ls = subprocess.run('mc ls g1/prod-landsat-ard2/', shell=True, capture_output=True, text=True, check=True).stdout
    tiles = re.findall(r'.\w{3}_\w{3}', ls, flags=re.MULTILINE)

    #%% Lets calculate how much time we need to load and prepare data for 1 tile
    #%%timeit -n 1 -r 1
    landsat_tile = '055W_06S' # Brazil # 007W_40N
    years = np.arange(2000, 2024)  # type: ignore
    samples_per_pixel = [1,2,4]  # 1 sample per point in time
    spps = []; times = []; spp=1
    fid = open('load_tiles_stats', 'w')
    fid.write('tile\tspp\ttime\n')
    notdone = True
    while notdone:
        #landsat_tile = np.random.choice(tiles)  # type: ignore   
        spp = np.random.choice(samples_per_pixel)  # type: ignore
        
        utils.ttprint(f"Loading tile {landsat_tile} with {spp} samples per pixel: ")
        start = time.time()
        landsat_files = utils.get_landsat_filenames_gaia(landsat_tile, years)
        landsat_data, crs, transform, bounds = utils.get_landsat_data(landsat_files, years)
        modis_data = utils.get_modis_ndvi_data_rio(landsat_files, years, crs, bounds)
        utils.ttprint(f"Landsat + modis: {time.time() - start:.2f} seconds")

        # masking
        landsat_data = utils.mask_from_qa(landsat_data, len(years))
        landsat_data = utils.mask_from_modis(landsat_data, modis_data, len(years))

        # get covariates
        covariate_data, covariate_names = utils.get_dtm_covariates(landsat_files)

        # Prepare dataset
        # Skip for now, shouldnt take much time
        # But need to save it to zarr

        stop= time.time()
        eta = stop - start
        print(f"spp={spp}, time: {eta:.0f} seconds")
        fid.write(f"{landsat_tile}\t{spp}\t{eta:.0f}\n")
        fid.flush()
        notdone = False

    fid.close()

# 113 sec for landsat data, 79 sec for modis data, 216 secs for masking and dtm derivatives
# total 408 secs for lansat tile 007W_40N

# 094W_79N .. time: 283 seconds, but - it's empty (at least landsat_data)!!!!
# 134E_54N 325 secs, empty landsat data

# 055W_06S 747 secs, has data ...
def save_tile_data():
    import zarr
    start = time.time()
    root = zarr.group(f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}_all.zarr', overwrite=True)
    root.create_array('landsat_data', data=landsat_data)
    root.create_array('modis_data', data=modis_data)
    root.create_array('years', data=np.array(years))
    utils.ttprint(f"Saved masked Landsat data to zarr in {time.time() - start} seconds")
    #zarr.consolidate_metadata(f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}_all.zarr')

def load_tile_data():
#%%    
    landsat_tile = '055W_06S'
    start = time.time()
    utils.ttprint(f"Loading masked data from zarr for tile {landsat_tile}  ...")
    arrays = utils.load_from_zarr_parallel(f'/mnt/nibble/gen_cog/arcov2/landsat_masked_{landsat_tile}.zarr')
    landsat_data: NDArray[np.float32] = arrays['landsat_data']  # type: ignore
    modis_data: NDArray[np.float32] = arrays['modis_data']  # type: ignore
    years: NDArray[np.int32] = arrays['years']  # type: ignore 
    utils.ttprint(f"Loaded data from zarr in {time.time() - start} seconds")
    # Loaded data from zarr in 166.3748698234558 seconds

#%%
def prepare_dataset_landsat(landsat_data, years, modis_data, covariate_data, covariate_names, transform, number_of_pixels, samples_per_pixel):
    """
    Prepare dataset from Landsat data.
    :param landsat_data: Landsat data as a numpy array.
    :param transform: Affine transformation for the Landsat data.
    :param number_of_pixels: Number of pixels to sample.
    :param samples_per_pixel: Number of samples per pixel.
    :return: Prepared dataset as a pandas DataFrame.
    """
    ## number_of_pixels=10000; samples_per_pixel=100
    nyears = len(years)
    ndates = nyears * utils.n_imag_per_year

    dates, _, doy = utils.get_dates_doy(years)

    # Sample random pixels
    random_indices = np.random.choice(landsat_data.shape[1], 2*number_of_pixels, replace=False)
    
    # take only valid pixels
    valid_pixels = np.isfinite(landsat_data[:ndates, random_indices]).sum(axis=0) > samples_per_pixel # Check if we have enough pixels
    valid_pixels = random_indices[valid_pixels][:number_of_pixels]  # Take only the first number_of_pixels valid indices
    
def cfc_testing():
#%%
    import torch
    import torch.nn as nn
    import pytorch_lightning as pl    
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import Callback
    from cfc_model.torch_cfc import Cfc #, CFCModel, CFCDataModule

    import numpy as np

    # IDEJA:
    # Napraviti model sa 23 CfC-ova, kao ovdje: https://www.mdpi.com/sensors/sensors-25-01622/article_deploy/html/images/sensors-25-01622-g006-550.jpg
    # Jedan case je jedna cijela godina
    # Svaki CfC ima za ulaz podatke iz jednog image-a (datuma), izlaz ide dalje u sljedeći CfC, ali što bi onda predstavljao zadnji?
    # Možda da se iz svakog CfC-a izlaz uspoređuje sa točnom vrijednošću (što ako je missing?) i da se u svaki CfC dovedu 
    # outputi od prethodnog i sljedećeg datuma? I onda imamo dva forward runa i jedan backward run.
    # Možda da svaki CfC ima jedan switch koji je -1 ako dobiva podatke od prethodnog i 1 ako dobiva od sljedećeg
    # i dva kompatibilna ulaza - jedan za trenutni datum, a drugi za prethodni ili sljedeći (ovisno o switchu)
    # Prvi i zadnji nemoraju imati sve ulaze!!! Mogu biti drugačiji od srednjih.

    class TestLearner(pl.LightningModule):
        def __init__(self, model, hparams):
            super().__init__()
            self.model = model
            self.loss_fn = nn.MSELoss()
            self._hparams = hparams

        def _prepare_batch(self, batch):
            _, t, x, mask, y = batch

            '''
            t_elapsed = t[:, 1:] - t[:, :-1]
            t_fill = torch.zeros(t.size(0), 1, device=x.device)
            t = torch.cat((t_fill, t_elapsed), dim=1)

            t = t * self._hparams["tau"]
            '''
            # here we can do some data manipulation ...
            return x, t, mask, y

        def training_step(self, batch, batch_idx):
            x, t, mask, y = self._prepare_batch(batch)
        
            y_hat = self.model.forward(x, t, mask=mask)

            enable_signal = torch.sum(y, -1) > 0.0

            y_hat = y_hat[enable_signal]
            y = y[enable_signal]


            y = torch.argmax(y.detach(), dim=-1)
            loss = self.loss_fn(y_hat, y)

            preds = torch.argmax(y_hat.detach(), dim=-1)  # labels are given as one-hot
            acc = (preds == y).float().mean()
            self.log("train_acc", acc, prog_bar=True)
            self.log("train_loss", loss, prog_bar=True)
            return {"loss": loss}

    model = Cfc(
        in_features = 
    )
    CFC_MIXED = {
        "epochs": 100,
        "clipnorm": 0,
        "hidden_size": 256,
        "base_lr": 0.0005,
        "decay_lr": 0.99,
        "backbone_activation": "gelu",
        "backbone_units": 128,
        "backbone_layers": 2,
        "backbone_dr": 0.5,
        "weight_decay": 4e-05,
        "tau": 10,
        "batch_size": 64,
        "optim": "adamw",
        "init": 1.35,
        "use_mixed": True,
        "no_gate": False,
        "minimal": False,
    }
#%% Results

if __name__ == "__main__":
    #tile_load_stas()