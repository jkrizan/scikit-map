#%%
#import settings

import torch
import torch.nn  as nn
from torch.optim import Adam
import torchmetrics

import ray
import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig
import lightning.pytorch as pl

from pathlib import Path

import numpy as np
from cfc import CfcModel, ArcoV2Dataset, MemoryDataLoader

fld = Path(__file__).parent / "final" 

INDICES = ['ndvi']
OUTPUT_SIZE = len(INDICES)
INPUT_SIZE = len(INDICES) + 2
TIMELESS_SIZE = 3
SEQUENCE_LENGTH = 12
YEARS = range(2000, 2024)
FN_ZARR = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
LIMIT = 10 #500
PERCENT_PIXELS = 0.1 #0.07
BATCHSIZE = 1024
EPOCHS = 100
criterion = nn.MSELoss()

#%%
def load_dataset():
    dataset = ArcoV2Dataset(
        zarr_path=FN_ZARR,
        years=YEARS,
        indices=INDICES,
        limit=LIMIT,
        percent_pixels=PERCENT_PIXELS,
        sequence_length=SEQUENCE_LENGTH,
        return_tensors=True,
        device=None
    )
    dataset.prepare_all_cases()
    return dataset

def train_func(config):

    model = CfcModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        timeless_input_size=TIMELESS_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        backbone_layers = [config["n_backbone_size"]] * config["n_backbone_layers"],        
        hidden_size=config["hidden_size"],
    )
    model = ray.train.torch.prepare_model(model)
    device = model.device
    optimizer = Adam(model.parameters(), lr=config["lr"])

    # Data
    batchsize = config.get("batch_size", BATCHSIZE)
    ds: ArcoV2Dataset = ray.get(config["dataset"])
    # #################################################
    # Have to put dataset on each worker separately
    # But then all workers will have their own copy of dataset in memory
    # And they will use same data unless we do something about it
    # #################################################
    
    print(f"Using device: {device}, length of dataset: {len(ds)}")

    train_subset, val_subset = ds.get_train_validation_subset(ncases_validation=0.2)

    train_loader = MemoryDataLoader(ds, batch_size=batchsize, indexes=train_subset.indices, cache_length=100)
    val_loader = MemoryDataLoader(ds, batch_size=batchsize, indexes=val_subset.indices, cache_length=100)


    criterion = nn.MSELoss().to(device)
    for epoch in range(EPOCHS):
        model.train()
        rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
        for (y, x, tl, ts) in train_loader:
            optimizer.zero_grad()
            outputs = model(x, tl, ts)
            rmse.update(outputs.detach(), y)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        avg_train_loss = rmse.compute().item()

        model.eval()
        with torch.no_grad():
            rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
            for (y, x, tl, ts) in val_loader:
                outputs = model(x, tl, ts)
                #loss = criterion(outputs, y)
                rmse.update(outputs, y)

        avg_val_loss = rmse.compute().item()

        # Log metrics
        ray.train.report({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch})


#%%
def train(config, dataset):

    run_config = RunConfig(storage_path = (fld / "ray_results").as_posix(), name="v1")
    scaling_config = ScalingConfig(num_workers=4, use_gpu=True, resources_per_worker={"CPU":2, "GPU": 1})

    trainer = TorchTrainer(train_func, 
                        train_loop_config = config,
                        scaling_config=scaling_config, 
                        run_config=run_config,
                        )
    result: ray.train.Result = trainer.fit()

def main():
    # LIMIT=10; PERCENT_PIXELS = 0.3
    ray.init(ignore_reinit_error=True, object_store_memory=300*1024*1024*1024)

    config = {
        "lr": 0.000215,
        "batch_size": 2048,  #tune.choice([2048, 4096, 8192]),       
        "max_num_epochs": 10,
        "n_backbone_layers": 3,
        "n_backbone_size": 72,
        "hidden_size": 80  
    }

    dataset = ArcoV2Dataset(
        zarr_path=FN_ZARR,
        years=YEARS,
        indices=INDICES,
        limit=LIMIT,
        percent_pixels=PERCENT_PIXELS,
        sequence_length=SEQUENCE_LENGTH,
        return_tensors=True,
        device='cuda'
    )
    dataset.prepare_all_cases()

    dsr = ray.put(dataset)
    config['dataset'] = dsr
    try: 
        train(config, dataset=dsr)
    finally:
        ray.shutdown()

    # ds_y = ray.data.from_numpy(dataset.all_y, override_num_blocks = 16)
    # ds_x = ray.data.from_numpy(dataset.all_x)
    # ds_x_gtemp = ray.data.from_numpy(dataset.all_x_gtemp)
    # ds_timespans = ray.data.from_numpy(dataset.all_timespans)    

    # ds = ray.data.from_items([dict(y=d[0], x=d[1], tl=d[2], ts=d[3]) for d in dataset]) # sporo


    #ds = ray.data.from_numpy()
    #ds = ray.data.from_items([dict(y=dataset.all_y, x=dataset.all_x, tl=dataset.all_x_gtemp, ts=dataset.all_timespans)]) # sporo
    # print(ds.schema())

    
if __name__ == "__main__":
    main()
