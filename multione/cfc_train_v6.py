#%%
#import ncps
from ast import List
from typing import Any
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import gc

#import utils, processing_utils
import time
import matplotlib.pyplot as plt
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import torch.nn as nn
import torch

#from cfc_dataset import ArcoV2DatasetV2
from cfc_v6 import CfcLearnerV6
import pickle

# Tensor cores:
# https://medium.com/@michael.diggin/the-power-of-8-getting-the-most-out-of-tensor-cores-c7704ae0c5c1

#%%


def train_v6(band, devices):
#%%
    # band = 1; devices=[0,1,2,3]
    input_size = 3 #8
    timeless_size = 3
    output_size = 1 #6 #7
    sequence_length = 12
    years = np.arange(2000, 2024)
    fn_zarr = Path(f"/home/josip/arcov2/sample_v6.zarr")
    limit = None
    percent_pixel=0.1

    learner = CfcLearnerV6(fn_zarr, 
                            years, 
                            input_size,                            
                            hidden_size=192, 
                            sequence_length=sequence_length,
                            timeless_input_size=timeless_size,
                            band = band, # nir 
                            output_size=output_size,
                            backbone_layers=[192,128,64,32,16,8],
                            limit=limit,
                            percent_pixels=percent_pixel,
                            device='cpu',
                            dtype=torch.float32,
                            batch_size=4096*2,
                            activation='relu',  ##silu, relu, tanh, gelu, lecun_tanh
                            lr=0.001,
                            debug=False)
#%% 
    '''
    learner.setup()
    y, x, timespans = next(iter(learner.train_dataloader()))
    y= y.to(torch.float32); x=x.to(torch.float32); timespans = timespans.to(torch.float32)
    loss = learner.training_step((y, x, timespans),0)
    '''

    #torch.multiprocessing.set_start_method('spawn')
    #torch.set_float32_matmul_precision('medium')
    checkpoint_callback = ModelCheckpoint(
        # dirpath=checkpoints_path, # <--- specify this on the trainer itself for version control
        filename="cfc_v6"+f"_b{band}"+"_{epoch:03d}",
        every_n_epochs=1,
        monitor='val_loss',
        save_top_k=5,  # <--- this is important!
        save_last = True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"cfc_v6_b{band}_last"
    checkpoint_callback.CHECKPOINT_NAME_BEST = f"cfc_v6_b{band}_best"
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "-"

    import os
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["TORCH_USE_CUDA_DSA"] = "1"

    trainer = pl.Trainer(max_epochs=100,
                         callbacks=[checkpoint_callback],
                         num_nodes=1, 
                         devices=devices,
                         #precision='16-mixed') #, devices=[0,1])
    )
    trainer.fit(learner) #, train_loader, val_loader)

#%%

#%%
def train_v6_continue():
    band = 1; devices=[0,1,2,3]
    input_size = 3 #8
    timeless_size = 3
    output_size = 1 #6 #7
    sequence_length = 12
    years = np.arange(2000, 2024)
    fn_zarr = Path(f"/home/josip/arcov2/sample_v6.zarr")
    limit = None
    percent_pixel=0.2

    learner = CfcLearnerV6(fn_zarr, 
                            years, 
                            input_size,                            
                            hidden_size=192, 
                            sequence_length=sequence_length,
                            timeless_input_size=timeless_size,
                            band = band, # nir 
                            output_size=output_size,
                            backbone_layers=[192,128,64,32,16,8],
                            limit=limit,
                            percent_pixels=percent_pixel,
                            device='cpu',
                            dtype=torch.float32,
                            batch_size=4096*2,
                            activation='relu',  ##silu, relu, tanh, gelu, lecun_tanh
                            lr=0.001,
                            debug=False)
    
    checkpoint_callback = ModelCheckpoint(
        # dirpath=checkpoints_path, # <--- specify this on the trainer itself for version control
        filename="cfc_v6"+f"_b{band}"+"_{epoch:03d}",
        every_n_epochs=1,
        monitor='val_loss',
        save_top_k=5,  # <--- this is important!
        save_last = True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"cfc_v6_b{band}_last"
    checkpoint_callback.CHECKPOINT_NAME_BEST = f"cfc_v6_b{band}_best"
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "-"

    import os
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["TORCH_USE_CUDA_DSA"] = "1"

    trainer = pl.Trainer(max_epochs=150,
                         callbacks=[checkpoint_callback],
                         num_nodes=1, 
                         devices=devices,
                         precision='16-mixed') #, devices=[0,1])

    torch.set_float32_matmul_precision('medium')
    
    trainer.fit(learner, ckpt_path="/home/josip/scikit-map/multione/lightning_logs/version_5/checkpoints/cfc_v6_b1_epoch-009.ckpt") #, train_loader, val_loader)

if __name__=="__main__":
    #train_v6_continue()
    
    import sys 
    band = int(sys.argv[1])
    devices = [int(x) for x in sys.argv[2:]]
    print(f"Training band {band} on devices {devices}")
    train_v6(band, devices)
    
    
    #train_test_v5_2_continue()
    #train_test_v5_1_continue()
    #train_test_v4_continue()
# %%
