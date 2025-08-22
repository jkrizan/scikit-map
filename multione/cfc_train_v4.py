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
import torchmetrics
import torch.nn as nn
import torch

from cfc_dataset import ArcoV2DatasetV2
from cfc_v4 import CfcModel_v4, CfcLearner_v4

fn_zarr = Path(f"/data/oemc/arcov2/sample_v1.zarr")
fn_zarr = Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.zarr")
years = np.arange(2000, 2024)

#%%

def train_test_v4():
    input_size = 9
    output_size = 7
    sequence_length=12

    learner = CfcLearner_v4(fn_zarr, 
                            years, 
                            input_size, 
                            hidden_size=128, 
                            sequence_length=sequence_length, 
                            output_size=output_size,
                            backbone_layers=[128,64,32],
                            limit=100, 
                            lr=0.1,
                            debug=True)

    torch.set_float32_matmul_precision('medium')
    trainer = pl.Trainer(max_epochs=100, num_nodes=1) #, devices=[0,1])
    trainer.fit(learner) #, train_loader, val_loader)



    

if __name__=="__main__":
    #train_test_v2()
    train_test_v4()
# %%
