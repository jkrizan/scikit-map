#%%
import glob
import pickle
import time
from typing import Dict, Optional, Any

import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import torchvision
import torchvision.transforms as transforms
from filelock import FileLock
from torch.utils.data import random_split

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler

from cfc_v9 import CfcLearnerV9, CfcModelV9, ArcoV2DatasetV9, MemoryDataLoader
import torch
import numpy as np
from pathlib import Path

global_trial_number = 0
#%%

fld_save_models = Path(__file__).parent / "raytuna_models_v9" 
fld_save_models.mkdir(exist_ok=True, parents=True)

DTYPE = torch.float32
BATCHSIZE = 4096*2    
#DEVICES = [0,1,2,3]
INDICES = ['ndvi']; 
OUTPUT_SIZE = len(INDICES)
INPUT_SIZE = len(INDICES) + 2
TIMELESS_SIZE = 3
SEQUENCE_LENGTH = 12
YEARS = np.arange(2000, 2024)
FN_ZARR = Path(f"/home/josip/arcov2/sample_v6.zarr")
LIMIT = None
PERCENT_PIXEL = 0.1
EPOCHS = 100
criterion = nn.MSELoss()

config = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "batch_size": tune.choice([2048, 4096, 8192]),    
    "num_trials": 100,
    "max_num_epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "n_backbone_layers": tune.choice([3, 4, 5]),
    "n_backbone_size": tune.choice([56, 64, 72, 80]),
    "hidden_size": tune.choice([56, 64, 72, 80]),
                                    
}

def load_data():
    with FileLock("data.lock"):
        dataset = ArcoV2DatasetV9( 
                                FN_ZARR,
                                years=YEARS,
                                sequence_length=SEQUENCE_LENGTH,
                                indices=INDICES,
                                limit=LIMIT,
                                percent_pixels=PERCENT_PIXEL,
                                device='cpu',
                                dtype=DTYPE)
        dataset.prepare_all_cases()

    return dataset

def create_dataloaders(dataset, batch_size, num_workers=8):

    train_subset, val_subset = dataset.get_train_validation_subset(ncases_validation=0.2)

    train_loader = MemoryDataLoader(dataset, batch_size=batch_size, indexes=train_subset.indices, cache_length=100)
    val_loader = MemoryDataLoader(dataset, batch_size=batch_size, indexes=val_subset.indices, cache_length=100)

    return train_loader, val_loader

def train_model(config):
    global global_trial_number
    trial_number = global_trial_number
    global_trial_number += 1

    pickle.dump(config, open(fld_save_models / f"{trial_number}_config.pkl", "wb"))
    print(f"Starting trial {trial_number} with config: {config}")
    
    model = CfcModelV9(input_size=INPUT_SIZE,
                       hidden_size=config["hidden_size"], 
                       timeless_input_size=TIMELESS_SIZE,
                       sequence_length=SEQUENCE_LENGTH, 
                       backbone_layers=[config["n_backbone_size"]] * config["n_backbone_layers"],
                       activation='lecun_tanh')
    device = config["device"]
    if device == "cuda":
        model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Load existing checkpoint through `get_checkpoint()` API.
    if tune.get_checkpoint():
        loaded_checkpoint = tune.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    # Data setup
    # if config["smoke_test"]:
    #     pass
    #     #trainset, _ = load_test_data()
    # else:
    dataset = load_data()

    train_loader, val_loader = create_dataloaders(
        dataset,
        config["batch_size"],
        num_workers=16 #if config["smoke_test"] else 8
    )

    best_val_loss = float('inf')
    for epoch in range(config["max_num_epochs"]):  # loop over the dataset multiple times
        model.train()

        for batch in train_loader:
            y,x,tl,ts = batch
            y,x,tl,ts = y.to(device), x.to(device), tl.to(device), ts.to(device)

            # forward + backward + optimize
            optimizer.zero_grad()  # reset gradients
            outputs = model(x, tl, ts)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()                        

        # Validation
        model.eval()
        val_loss = 0.0
        correct = total = 0
        with torch.no_grad():
            rmse = torchmetrics.MeanSquaredError(squared=True).to(device)
            for batch in val_loader:
                y,x,tl,ts = batch
                y,x,tl,ts = y.to(device), x.to(device), tl.to(device), ts.to(device)
                outputs = model(x, tl, ts)
                rmse.update(outputs, y)

        loss = rmse.compute().item()
        if loss < best_val_loss:
            best_val_loss = loss
            path = os.path.join(fld_save_models, f"{trial_number}_best.pt")
            torch.save(model.state_dict(), path)
            print(f"Trial {trial_number}, epoch {epoch}, new best validation loss={best_val_loss:.6f} ")

        # Report metrics
        metrics = {
            "loss": loss,
        }

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
        # in future iterations.
        # Note to save a file-like checkpoint, you still need to put it under a directory
        # to construct a checkpoint.
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = tune.Checkpoint.from_directory(temp_checkpoint_dir)
            tune.report(metrics, checkpoint=checkpoint)
    print(f"Finished Training {trial_number}!")

def main(config, gpus_per_trial=0.5):
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=config["max_num_epochs"],
        grace_period=1,
        reduction_factor=2)
    
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model),
            resources={"cpu": 1, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=config["num_trials"],
        ),
        param_space=config,
    )
    results = tuner.fit()
    df = results.get_dataframe()
    df.to_csv(fld_save_models / "tune_results.csv",sep='\t')
    best_result = results.get_best_result("loss", "min")
    

    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.metrics['loss']}")
 
    #test_best_model(best_result, smoke_test=config["smoke_test"])

main(config, gpus_per_trial=0.25 if torch.cuda.is_available() else 0)