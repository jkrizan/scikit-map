#%%
#import ncps
from ast import List
import fnmatch
from turtle import back
from typing import Any
from xml.sax.handler import DTDHandler
from minio.lifecycleconfig import G
from minio.xml import B
from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import gc
import warnings

import torchmetrics
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas
from sklearn.calibration import Hidden

from utils import ttprint
#import utils, processing_utils
import time
import matplotlib.pyplot as plt
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback

import torch.nn as nn
import torch

#from cfc_dataset import ArcoV2DatasetV2
from cfc_v8 import CfcLearnerV8, CfcModelV8, ArcoV2DatasetV8, MemoryDataLoader
import pickle

from torch.utils.data import DataLoader
import optuna
from optuna import Trial, visualization
import torch.optim as optim
import logging
import sys
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from contextlib import contextmanager
import multiprocessing
N_GPUS = 4

from torch.profiler import profile, ProfilerActivity


# Tensor cores:
# https://medium.com/@michael.diggin/the-power-of-8-getting-the-most-out-of-tensor-cores-c7704ae0c5c1

#%%

class TuningCallback(Callback):
    def __init__(self):
        super().__init__()
        self.best_val_loss = float('inf')

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        val_loss = trainer.callback_metrics.get("val_loss")
        trial: Trial = pl_module.trial
        trial.report(val_loss.item(), step=trainer.current_epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
class GpuQueue:

    def __init__(self):
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = list(range(N_GPUS)) if N_GPUS > 0 else [None]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)
        

class Objective:

    def __init__(self, gpu_queue: GpuQueue, fld_save_models: Path):
        self.gpu_queue = gpu_queue
        self.fld_save_models = fld_save_models

        self.DTYPE = torch.float16
        self.BATCHSIZE = 4096*2    
        #DEVICES = [0,1,2,3]
        self.BANDS = [0,1,2,3,4,5]; 
        self.OUTPUT_SIZE = len(self.BANDS)
        self.INPUT_SIZE = len(self.BANDS) + 2
        self.TIMELESS_SIZE = 3
        self.SEQUENCE_LENGTH = 12
        self.YEARS = np.arange(2000, 2024)
        self.FN_ZARR = Path(f"/home/josip/arcov2/sample_v6.zarr")
        self.LIMIT = None
        self.PERCENT_PIXEL = 0.2
        self.EPOCHS = 100
        self.criterion = nn.MSELoss()

        self.datasets = {}
        self.subsets = {}

        self.dataset = ArcoV2DatasetV8( self.FN_ZARR,
                        years=self.YEARS,
                        sequence_length=self.SEQUENCE_LENGTH,
                        bands=self.BANDS,
                        limit=self.LIMIT,
                        percent_pixels=self.PERCENT_PIXEL,
                        device='cpu',
                        dtype=self.DTYPE
        )
        self.dataset.prepare_all_cases()

        # self.ds = ds
        # self.train_subset, self.valid_subset = ds.get_train_validation_subset(0.2)

        # self.train_loader = DataLoader(train_subset, batch_size=self.BATCHSIZE, persistent_workers=True,
        #                                         shuffle=True, num_workers=4, prefetch_factor=4)
        # self.val_loader = DataLoader(valid_subset, batch_size=self.BATCHSIZE, persistent_workers=True,
        #                                         shuffle=False, num_workers=4, prefetch_factor=4)
        

    def get_train_val_loaders(self, gpu_i: int) -> tuple[MemoryDataLoader, MemoryDataLoader]:
        ttprint(f"Getting data loaders for GPU {gpu_i}")
        device = f'cuda:{gpu_i}'
        if gpu_i not in self.datasets:
            
            ttprint(f"Cloning dataset to GPU {gpu_i}")
            self.datasets[gpu_i] = self.dataset.clone_to(device)
            ttprint(f"Dataset cloned to GPU {gpu_i}")

            #self.datasets[gpu_i].prepare_all_cases()
            train_subset, valid_subset = self.datasets[gpu_i].get_train_validation_subset(0.2)                        
            #train_subset = torch.tensor(train_subset).to(device)
            #valid_subset = torch.tensor(valid_subset).to(device)
            self.subsets[gpu_i] = (train_subset, valid_subset)
            ttprint(f"Subsets prepared on GPU {gpu_i}")

        train_subset, valid_subset = self.subsets[gpu_i]
        # train_loader = DataLoader(train_subset, batch_size=self.BATCHSIZE, persistent_workers=False,
        #                                     shuffle=True, num_workers=0)
        # val_loader = DataLoader(valid_subset, batch_size=self.BATCHSIZE, persistent_workers=False,
        #                                     shuffle=False, num_workers=0)
        train_inds = torch.tensor(train_subset.indices).to(device)
        valid_inds = torch.tensor(valid_subset.indices).to(device)

        train_loader = MemoryDataLoader(self.datasets[gpu_i], batch_size=self.BATCHSIZE, indexes=train_inds)
        val_loader = MemoryDataLoader(self.datasets[gpu_i], batch_size=self.BATCHSIZE, indexes=valid_inds)

        ttprint(f"Data loaders ready on GPU {gpu_i}")
        return train_loader, val_loader
    

    def __call__(self, trial: Trial):
        with self.gpu_queue.one_gpu_per_process() as gpu_i:
            DEVICE = f'cuda:{gpu_i}'  #'cuda:0' # 'cpu' # 'cuda:0' #

            hidden_size = trial.suggest_categorical("hidden_size", [64, 72, 80])
            #batch_size = trial.suggest_categorical("batch_size", [2048, 4096, 8192])
            activation = 'lecun_tanh'  # trial.suggest_categorical("activation", ['tanh', 'lecun_tanh'])    # ['relu', 'silu', 'gelu', 'tanh', 'lecun_tanh']
            lr = trial.suggest_float("lr", 0.0001, 0.001, log=False)
            n_backbone_layers = trial.suggest_int("n_backbone_layers", 4, 5)
            max_backbone_layer_size = 80 #(2 ** n_backbone_layers ) * 6 # max = 256
            min_backbone_layer_size = 64 #(2 ** (n_backbone_layers - 1)) * 6  # min = 128
            backbone_layer_size = trial.suggest_int("backbone_layer_size", min_backbone_layer_size, max_backbone_layer_size, step=8)
            backbone_layers = [backbone_layer_size] * n_backbone_layers #[first_backbone_layer_size // (2 ** i) for i in range(n_backbone_layers)]
            #backbone_dropout = 0 # trial.suggest_categorical("backbone_dropout", [0.0, 0.01, 0.02, 0.04, 0.08, 0.1])
            
            print(f"Trial {trial.number}: hidden_size={hidden_size}, lr={lr}, activation={activation}, Backbone layers: {backbone_layers}")             
            print()

            model = CfcModelV8(input_size=self.INPUT_SIZE,
                                hidden_size=hidden_size, 
                                timeless_input_size=self.TIMELESS_SIZE,
                                sequence_length=self.SEQUENCE_LENGTH, 
                                output_size=self.OUTPUT_SIZE, 
                                backbone_layers=backbone_layers, 
                                backbone_dropout=0.0,
                                activation=activation).to(device=DEVICE, dtype=self.DTYPE)

            optimizer_name = "Adam" # trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
            optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

            train_loader, val_loader = self.get_train_val_loaders(gpu_i)
            # train_loader = DataLoader(self.train_subset, batch_size=self.BATCHSIZE, persistent_workers=True,
            #                                     shuffle=True, num_workers=4, prefetch_factor=4)
            # val_loader = DataLoader(self.valid_subset, batch_size=self.BATCHSIZE, persistent_workers=True,
            #                                     shuffle=False, num_workers=4, prefetch_factor=4)
            best_val_loss = float('inf')
            for epoch in range(self.EPOCHS):
                ttprint(f"GPU {gpu_i}, Trial {trial.number}, Epoch {epoch}")
                model.train()                
                for i,batch in enumerate(train_loader):
                    #if (i%100)==0:
                    #    ttprint(f"GPU {gpu_i}, Trial {trial.number}, Epoch {epoch}, Batch {i}")
                    (y, x, timeless, timespans) = batch
                    optimizer.zero_grad()
                    outputs = model(x, timeless, timespans)
                    loss = self.criterion(outputs, y)
                    loss.backward()
                    optimizer.step()


                model.eval()
                with torch.no_grad():
                    rmse = torchmetrics.MeanSquaredError(squared=True).to(DEVICE)
                    for batch in val_loader:                
                        (y, x, timeless, timespans) = batch
                        output = model(x, timeless, timespans)
                        rmse.update(output, y)                        

                loss = rmse.compute().item()
                if loss < best_val_loss:
                    best_val_loss = loss
                    torch.save(model.state_dict(), self.fld_save_models / f"trial_{trial.number}_best_model.pth")
                    print(f"Trial {trial.number}, epoch {epoch}, new best validation loss={best_val_loss:.6f} ")

                trial.report(loss, epoch)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return loss


def train_v8_hptuning():
    study_name = "cfc_v8_opt_3"
    fld_save_models = Path(__file__).parent / "optuna_models_3"/ study_name
    fld_save_models.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(storage="sqlite:///cfc_v8_opt.sqlite3", direction="minimize", study_name=study_name, load_if_exists=True)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study.optimize(Objective(GpuQueue(), fld_save_models), n_trials=1000, timeout=None, n_jobs=32)   
    ### !!!!!!!! ####
    # PROBLEM: After all jobs are done, the script does not terminate. It just hangs.
    # Possible reason: Multiprocessing queue does not close properly?
    ### !!!!!!!! ####

    # obj = Objective(GpuQueue())
def hptzuning_export():   
    INPUT_SIZE=8
    TIMELESS_SIZE=3
    SEQUENCE_LENGTH=12
    OUTPUT_SIZE=6
    DEVICE = 'cpu'
    activation = 'lecun_tanh'

    study_name = "cfc_v8_opt_2"
    fld_save_models = Path(__file__).parent / "optuna_models"/ study_name
    study = optuna.create_study(storage="sqlite:///cfc_v8_opt.sqlite3", direction="minimize", study_name=study_name, load_if_exists=True)

    # Export the study
    df = study.trials_dataframe() #.to_csv(fld_save_models / "trials.csv", index=False)
    df = df.dropna()
    df = df.sort_values(by="value")

    for i, row in df.iterrows():
        # i=0; row = df.iloc[i]
        hidden_size = int(row['params_hidden_size'])
        backbone_layers = [int(row['params_backbone_layer_size'])] * int(row['params_n_backbone_layers'])
        print(f"{i} Trial {row['number']}: Value={row['value']}, hidden_size={hidden_size}, Backbone layers: {backbone_layers}")
        model_path = fld_save_models / f"trial_{row['number']}_best_model.pth"
        
        model = CfcModelV8(input_size=INPUT_SIZE,
                                hidden_size=hidden_size, 
                                timeless_input_size=TIMELESS_SIZE,
                                sequence_length=SEQUENCE_LENGTH, 
                                output_size=OUTPUT_SIZE, 
                                backbone_layers=backbone_layers, 
                                backbone_dropout=0.0,
                                activation=activation).to(DEVICE)
        total_params = sum(p.numel() for p in model.parameters())
        df.at[i, 'total_params'] = total_params
        print(f"Total parameters: {total_params}")
        #model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    df.to_csv(fld_save_models / "trials.csv", index=False, sep='\t')
    print("Trials saved to CSV")

def train_v8_hptuning_continue():
    import pandas
    study_name = "cfc_v8_opt_2"
    fld_save_models = Path(__file__).parent / "optuna_models"/ study_name    

    df = pandas.read_csv(fld_save_models / "trials.csv", sep='\t')
    df['rmse'] = np.sqrt(df['value'])
    df['rmse_times_params'] = df['rmse'] * df['total_params']
    df = df.sort_values(by="rmse_times_params")
    df = df[df['rmse_times_params'] < 4000]  # 0.2
    print("Filtered trials: ", len(df))

    for i, row in df.iterrows():        
        #row = df.loc[i]
        hidden_size = int(row['params_hidden_size'])
        backbone_layers = [int(row['params_backbone_layer_size'])] * int(row['params_n_backbone_layers'])
        lr = float(row['params_lr']) # 0.000247
        lr = lr/10
        print(f"{i} Trial {row['number']}: Value={row['value']}, RMSE={row['rmse']}, RMSE*Params={row['rmse_times_params']}, hidden_size={row['params_hidden_size']}, Backbone layers: {backbone_layers}")

        model_path = fld_save_models / f"trial_{row['number']}_best_model.pth"
        INPUT_SIZE=8
        TIMELESS_SIZE=3
        SEQUENCE_LENGTH=12
        OUTPUT_SIZE=6
        DEVICE = 'cpu'
        percent_pixel=0.5
        years = np.arange(2000, 2024)
        bands = [0,1,2,3,4,5]
        activation = 'lecun_tanh'
        learner = CfcLearnerV8( Path(f"/home/josip/arcov2/sample_v6.zarr"), 
                                years, 
                                INPUT_SIZE,                            
                                hidden_size=hidden_size,
                                sequence_length=SEQUENCE_LENGTH,
                                timeless_input_size=TIMELESS_SIZE,
                                bands=bands, # nir                             
                                backbone_layers=backbone_layers,
                                limit=None,
                                percent_pixels=percent_pixel,
                                device='cpu',
                                dtype=torch.float32,
                                batch_size=4096*2,
                                activation=activation,  ##silu, relu, tanh, gelu, lecun_tanh
                                lr=0.001,
                                debug=False)
        learner.model.load_state_dict(torch.load(model_path))

        checkpoint_callback = ModelCheckpoint(
            # dirpath=checkpoints_path, # <--- specify this on the trainer itself for version control
            filename=f"v8_{row['number']}" + "_{epoch:03d}",
            every_n_epochs=1,
            monitor='val_loss',
            save_top_k=5,  # <--- this is important!
            save_last=False
        )
        early_stopping_callback = EarlyStopping('val_loss', patience=5, verbose=True, mode='min')

        trainer = pl.Trainer(max_epochs=200,
                            strategy = 'ddp_find_unused_parameters_true',
                            callbacks=[checkpoint_callback, early_stopping_callback],
                            num_nodes=1, 
                            devices=[0,1,2,3],
                            
                            #precision='16-mixed') #, devices=[0,1])
        )
        trainer.fit(learner) 

#%%
def train_v8_test():

    devices=[0,1,2,3]    
    bands = [0,1,2,3,4,5]; 
    output_size = len(bands)
    input_size = len(bands) + 2
    timeless_size = 3
    sequence_length = 12
    years = np.arange(2000, 2024)
    fn_zarr = Path(f"/home/josip/arcov2/sample_v6.zarr")
    limit = 20
    percent_pixel=0.1
    device = f'cuda:0'

    ds = ArcoV2DatasetV8(fn_zarr,
                        years=years,
                        sequence_length=sequence_length,
                        bands=bands,
                        limit=limit,
                        percent_pixels=percent_pixel,
                        device='cpu',
                        dtype=torch.float32
        )
    ds.prepare_all_cases()

    ds0 = ds.clone_to(device)
    train_loader = MemoryDataLoader(ds0, batch_size=4096, indexes=torch.tensor(np.arange(0, len(ds0))).to(device))

    model = CfcModelV8(input_size=input_size,
                                hidden_size=96, 
                                timeless_input_size=timeless_size,
                                sequence_length=sequence_length, 
                                output_size=output_size, 
                                backbone_layers=[64, 64, 64], 
                                backbone_dropout=0,
                                activation='lecun_tanh').to(device)
    optimizer = getattr(optim, 'Adam')(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(10):
        # epoch = 0
        ttprint(f"Epoch {epoch}")
        model.train()                
        for i,batch in enumerate(train_loader):
            # i = 0; batch = next(iter(train_loader))
            #if (i%100)==0:
            #    ttprint(f"GPU {gpu_i}, Trial {trial.number}, Epoch {epoch}, Batch {i}")
            (y, x, timeless, timespans) = batch
            optimizer.zero_grad()
            outputs = model(x, timeless, timespans)
            loss = criterion(outputs, y)
            print(f"Batch {i}, loss={loss.item()}")
            loss.backward()
            optimizer.step()



def train_v8(devices):
    import random

    #devices=[0,1,2,3]
    bands = [0,1,2,3,4,5]; 
    output_size = len(bands)
    input_size = len(bands) + 2
    timeless_size = 3
    sequence_length = 12
    years = np.arange(2000, 2024)
    fn_zarr = Path(f"/home/josip/arcov2/sample_v6.zarr")
    limit = None
    percent_pixel=0.2
    hidden_size = 64 # random.choice([56, 64, 72, 80])
    backbone_size = 72 # random.choice([56, 64, 72, 80])
    backbone_layers=[backbone_size] * 5
    lr = 0.0005 # * random.randrange(2, 20, 2)

    learner = CfcLearnerV8(fn_zarr, 
                            years, 
                            input_size,                            
                            hidden_size=hidden_size, 
                            sequence_length=sequence_length,
                            timeless_input_size=timeless_size,
                            bands=bands, # nir                             
                            backbone_layers=backbone_layers,
                            limit=limit,
                            percent_pixels=percent_pixel,
                            device='cpu',
                            dtype=torch.float32,
                            batch_size=4096*2,
                            activation='lecun_tanh',  ##silu, relu, tanh, gelu, lecun_tanh
                            lr=lr,
                            debug=False)

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
        filename="v2" + "_{epoch:03d}",        
        every_n_epochs=1,
        monitor='val_loss',
        save_top_k=3,  # <--- this is important!
        save_last=False
    )
    #checkpoint_callback.CHECKPOINT_NAME_LAST = f"cfc_v8_last"
    #checkpoint_callback.CHECKPOINT_NAME_BEST = f"cfc_v8_b{band}_best"
    #checkpoint_callback.CHECKPOINT_EQUALS_CHAR = "-"

    early_stopping_callback = EarlyStopping('val_loss', patience=5, verbose=True, mode='min')

    #import os
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["TORCH_USE_CUDA_DSA"] = "1"

    trainer = pl.Trainer(max_epochs=100,
                         callbacks=[checkpoint_callback, early_stopping_callback],
                         strategy = 'ddp_find_unused_parameters_true',
                         num_nodes=1, 
                         devices=devices,
                         #precision='16-mixed') #, devices=[0,1])
    )
    trainer.fit(learner) #, train_loader, val_loader)

#%%


if __name__=="__main__":
    while True:
        train_v8([0,1,2,3])

    #train_v8_hptuning()
    #train_v8_hptuning_continue()

    
    # import sys 
    # band = int(sys.argv[1])
    # devices = [int(x) for x in sys.argv[2:]]
    #    print(f"Training band {band} on devices {devices}")
    # train_v6(band, devices)
    
    
    # import sys
    # checkpoint_path = sys.argv[1]
    # print(f"Continuing training from checkpoint {checkpoint_path}")
    # train_v6_continue(checkpoint_path)

    #train_test_v5_2_continue()
    #train_test_v5_1_continue()
    #train_test_v4_continue()
# %%
