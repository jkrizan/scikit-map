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
from cfc_v5_1 import CfcModel_v5, CfcLearner_v5

# Tensor cores:
# https://medium.com/@michael.diggin/the-power-of-8-getting-the-most-out-of-tensor-cores-c7704ae0c5c1

#%%

def train_test_v5():
    input_size = 9
    output_size = 7
    sequence_length=12
    years = np.arange(2000, 2024)
    fn_zarr = Path(f"/home/josip/arcov2/sample_v1.zarr")

    learner = CfcLearner_v5(fn_zarr, 
                            years, 
                            input_size, 
                            hidden_size=128, 
                            sequence_length=sequence_length, 
                            output_size=output_size,
                            backbone_layers=[128,64,32],
                            limit=[200,400], 
                            activation='relu',  ##silu, relu, tanh, gelu, lecun_tanh
                            lr=0.01,
                            debug=False)

    #torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')
    checkpoint_callback = ModelCheckpoint(
        # dirpath=checkpoints_path, # <--- specify this on the trainer itself for version control
        filename="cfc_v5_e{epoch:02d}.ckpt",
        every_n_epochs=1,
        save_top_k=-1,  # <--- this is important!
    )
    trainer = pl.Trainer(max_epochs=100,
                         callbacks=[checkpoint_callback],
                         num_nodes=1) #, devices=[0,1])
    trainer.fit(learner) #, train_loader, val_loader)

def train_test_v5_1():
    input_size = 8 #9
    output_size = 6 #7
    sequence_length=12
    years = np.arange(2000, 2024)
    fn_zarr = Path(f"/home/josip/arcov2/sample_v1.zarr")

    learner = CfcLearner_v5(fn_zarr, 
                            years, 
                            input_size, 
                            hidden_size=128, 
                            sequence_length=sequence_length, 
                            output_size=output_size,
                            backbone_layers=[64,32,16],
                            limit=[200,400], 
                            activation='relu',  ##silu, relu, tanh, gelu, lecun_tanh
                            lr=0.01,
                            debug=False)

    #torch.multiprocessing.set_start_method('spawn')
    #torch.set_float32_matmul_precision('medium')
    checkpoint_callback = ModelCheckpoint(
        # dirpath=checkpoints_path, # <--- specify this on the trainer itself for version control
        filename="cfc_v5.1_e{epoch:03d}.ckpt",
        every_n_epochs=1,
        save_top_k=-1,  # <--- this is important!
    )
    trainer = pl.Trainer(max_epochs=100,
                         callbacks=[checkpoint_callback],
                         num_nodes=1, devices=[1,2,3],
                         precision='16-mixed') #, devices=[0,1])
    trainer.fit(learner) #, train_loader, val_loader)


    # epoch 7, step 16455, MSEtrain: 0.000408 MSEvalid: 0.000448
    # epoch 35, step 74051, MSEtrain:0.000408, MSEvalid: 0.00748

def train_test_v5_1_continue():
    input_size = 8 #9
    output_size = 6 #7
    sequence_length=12
    years = np.arange(2000, 2024)
    fn_zarr = Path(f"/home/josip/arcov2/sample_v1.zarr")

    learner = CfcLearner_v5(fn_zarr, 
                            years, 
                            input_size, 
                            hidden_size=128, 
                            sequence_length=sequence_length, 
                            output_size=output_size,
                            backbone_layers=[64,32,16],
                            limit=200, 
                            activation='relu',  ##silu, relu, tanh, gelu, lecun_tanh
                            lr=0.001,
                            debug=False)
    
    checkpoint_callback = ModelCheckpoint(
    # dirpath=checkpoints_path, # <--- specify this on the trainer itself for version control
    filename="cfc_v5.1c_e{epoch:03d}",
    every_n_epochs=1,
    save_top_k=-1,  # <--- this is important!
    )
    trainer = pl.Trainer(max_epochs=100,
                         callbacks=[checkpoint_callback],
                         num_nodes=1, devices=[1,2,3],
                         precision='16-mixed') #, devices=[0,1])
    
    trainer.fit(learner, ckpt_path="/home/josip/scikit-map/multione/lightning_logs/version_21/checkpoints/cfc_v5.1_eepoch=035.ckpt.ckpt") 

def train_test_v4_continue():
    input_size = 9
    output_size = 7
    sequence_length=12
    years = np.arange(2000, 2024)
    fn_zarr = Path(f"/home/josip/arcov2/sample_v1.zarr")

    learner = CfcLearner_v4(fn_zarr, 
                            years, 
                            input_size, 
                            hidden_size=128, 
                            sequence_length=sequence_length, 
                            output_size=output_size,
                            backbone_layers=[128,128,128],
                            limit=-100, 
                            activation='relu',  ##silu, relu, tanh, gelu, lecun_tanh
                            lr=0.0001,
                            debug=False)

    #torch.multiprocessing.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(max_epochs=150, num_nodes=1) 
    # benchmark=True - speedup if input size doesn't change
    # fast_dev_run = 1,2,3 - limit to 1,2,3 batches for debugging
    # reload_dataloaders_every_n_epochs  -- reloads training and validation dataloaders
    
    trainer.fit(learner, ckpt_path="/home/josip/scikit-map/multione/lightning_logs/version_22/checkpoints/cfc-v4_e-99.ckpt") #, train_loader, val_loader)

if __name__=="__main__":
    train_test_v5_1_continue()
    #train_test_v4_continue()
# %%
