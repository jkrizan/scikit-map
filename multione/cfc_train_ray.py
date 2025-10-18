#%%
import tempfile

from ray import tune
import settings
from ray.data import DataIterator
from ray.data.datasource import SaveMode
from sqlalchemy.engine import row
import torch
import torch.nn  as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchmetrics

import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint, ScalingConfig, RunConfig
from pathlib import Path
import ray
import numpy as np
from cfc import CfcModel, ArcoV2Dataset, MemoryDataLoader

fld = Path(__file__).parent / "final" 
storage_path = (fld / "ray_results").as_posix()

INDICES = ['fpar']
OUTPUT_SIZE = len(INDICES)
INPUT_SIZE = len(INDICES) + 2
TIMELESS_SIZE = 3
SEQUENCE_LENGTH = 12
YEARS = range(2000, 2024)
FN_ZARR = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
LIMIT =  500
PERCENT_PIXELS = 0.07
BATCHSIZE = 1024
EPOCHS = 100
criterion = nn.MSELoss()

#%%
def train_func(config):

    context = ray.train.get_context()
    rank = context.get_world_rank()
    nranks = context.get_world_size()

    model = CfcModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        timeless_input_size=TIMELESS_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        backbone_layers = [config["n_backbone_size"]] * config["n_backbone_layers"],        
        hidden_size=config["hidden_size"],
    )
    model = ray.train.torch.prepare_model(model)
    device: torch.device = model.device # type: ignore

    optimizer = Adam(model.parameters(), lr=config["lr"])

    # Data
    batchsize = config.get("batch_size", BATCHSIZE) # // nranks   # per worker batch size, but then its slower!
    ds: ArcoV2Dataset = ray.get(config["dataset"])

    # Need to get only part of dataset for each worker
    ds, val_indices, train_indices = ds.prepare_for_ray_worker(rank, nranks, ncases_validation=0.2, device = device)

    print(f"Rank: {rank}, all: {len(ds)}, train: {len(train_indices)}, val: {len(val_indices)} device: {device}")  

    train_loader = MemoryDataLoader(ds, batch_size=batchsize, indexes=train_indices, cache_length=100000)
    val_loader = MemoryDataLoader(ds, batch_size=batchsize, indexes=val_indices, cache_length=10000)
    print (f"Rank: {rank}, Train loader batches: {len(train_loader)}, Val loader batches: {len(val_loader)}")
    

    criterion = nn.MSELoss().to(device)
    for epoch in range(EPOCHS):
        model.train()
        rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
        for i, (y, x, tl, ts) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(x, tl, ts)
            rmse.update(outputs.detach(), y)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            #if rank == 0:
            #    print(f"Epoch {epoch:03}, batch {i:03}: loss: {loss.item():.6f}",end='\r')

        avg_train_loss = rmse.compute().item()
        #if rank == 0:
        #    print(f"Epoch {epoch:03}: train RMSE: {avg_train_loss:.4f}")

        model.eval()
        with torch.no_grad():
            rmse = torchmetrics.MeanSquaredError(squared=False).to(device)
            for i, (y, x, tl, ts) in enumerate(val_loader):
                outputs = model(x, tl, ts)
                loss = criterion(outputs, y)
                rmse.update(outputs, y)
                #if rank == 0:
                #    print(f"Epoch {epoch:03}, val batch {i:03}: loss: {loss.item():.6f}",end='\r')

        avg_val_loss = rmse.compute().item()

        metrics = {"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch}

        # Checkpoint
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = Path(temp_checkpoint_dir) / "checkpoint.pt" 
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            # Example: Only the rank 0 worker uploads the checkpoint.
            if ray.train.get_context().get_world_rank() == 0:
                ray.train.report(metrics, checkpoint=checkpoint)
            else:
                ray.train.report(metrics, checkpoint=None)   


#%%
def train(config: dict):

    run_config = RunConfig(storage_path = storage_path, name=config["name"])
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
    
    dataset = ArcoV2Dataset(
        zarr_path=FN_ZARR,
        years=YEARS,
        indices=INDICES,
        limit=LIMIT,
        percent_pixels=PERCENT_PIXELS,
        sequence_length=SEQUENCE_LENGTH,
        return_tensors=True,
        device='cpu'
    )
    print(f"Dataset loaded with {len(dataset)} samples.")
    print("Preparing all cases...")
    dataset.prepare_all_cases()
    dsr = ray.put(dataset)


    default_config = {
        "lr": 0.000215,
        "batch_size": 2048,      
        "max_num_epochs": 100,
        "dataset": dsr
    }

    configs = []

    # test config
    # configs.append({
    #     "name": "v1_test",  
    #     "n_backbone_layers": 3,
    #     "n_backbone_size": 72,
    #     "hidden_size": 80  
    # })

    configs.append({
        "name": "v1_smallest",                
        "n_backbone_layers": 3,
        "n_backbone_size": 56,
        "hidden_size": 56
    })
    configs.append({
        "name": "v2_besttuned",                
        "n_backbone_layers": 3,
        "n_backbone_size": 72,
        "hidden_size": 80  
    })
    configs.append({
        "name": "v3_large",                
        "n_backbone_layers": 4,
        "n_backbone_size": 72,
        "hidden_size": 72  
    })
    configs.append({
        "name": "v4_larger",        
        "n_backbone_layers": 4,
        "n_backbone_size": 72,
        "hidden_size": 72  
    })
    configs.append({
        "name": "v5_largest",        
        "n_backbone_layers": 4,
        "n_backbone_size": 80,
        "hidden_size": 80  
    })

    try:
        for config in configs:
            config.update(default_config)
            train(config)
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
