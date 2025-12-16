#%%
from tabnanny import check
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

#fld = settings.PRODUCTION_FOLDER #Path(__file__).parent / "final" 
fld = Path("/root")
fld_ray_results = (fld / "ray_results")

INDICES = ['fpar']
OUTPUT_SIZE = len(INDICES)
INPUT_SIZE = len(INDICES) + 2
TIMELESS_SIZE = 3
SEQUENCE_LENGTH = 12
YEARS = range(2000, 2024)
FN_ZARR = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
LIMIT = 500
PERCENT_PIXELS = 0.02
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
    n_params = model.n_params
    print(f"Rank {rank}: Model initialized with {n_params} trainable parameters.")
    model = ray.train.torch.prepare_model(model)
    device: torch.device = model.device # type: ignore

    if "checkpoint_path" in config:
        checkpoint_path = config["checkpoint_path"]
        print(f"Rank {rank}: Loading checkpoint from {checkpoint_path}")
        checkpoint_data = torch.load(checkpoint_path + '/checkpoint.pt', map_location=device)
        model.load_state_dict(checkpoint_data[0])
        print(f"Rank {rank}: Checkpoint loaded.")

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

        metrics = {"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch, "n_params": n_params}

        # Checkpoint
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = Path(temp_checkpoint_dir) / "checkpoint.pt" 
            torch.save(
                # Not good, need to unwrap module while loading, all keys will be prefixed with 'module.'
                (model.state_dict(), optimizer.state_dict()), path
                # This is better:
                # model.module.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            # Example: Only the rank 0 worker uploads the checkpoint.
            if ray.train.get_context().get_world_rank() == 0:
                ray.train.report(metrics, checkpoint=checkpoint)
            else:
                ray.train.report(metrics, checkpoint=None)   


#%%
def train(config: dict):

    run_config = RunConfig(storage_path = fld_ray_results, name=config["name"])
    scaling_config = ScalingConfig(num_workers=4, use_gpu=True, resources_per_worker={"CPU":3, "GPU": 1})

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
        "lr": 0.0002,
        "batch_size": 2048,      
        "max_num_epochs": 100,
        "dataset": dsr
    }

    configs = []

    # test config
    configs.append({
        "name": "v0_xs_5_8_16",  
        "n_backbone_layers": 5,
        "n_backbone_size": 8,
        "hidden_size": 16  
    })
    configs.append({
        "name": "v0_xs_4_12_24", 
        "n_backbone_layers": 4,
        "n_backbone_size": 12,
        "hidden_size": 24  
    })
    configs.append({
        "name": "v0_xs_3_12_12", 
        "n_backbone_layers": 3,
        "n_backbone_size": 12,
        "hidden_size": 12  
    })
    configs.append({
        "name": "v0_xs_2_12_12", 
        "n_backbone_layers": 2,
        "n_backbone_size": 12,
        "hidden_size": 12  
    })
    configs.append({
        "name": "v0_xs_2_8_8", 
        "n_backbone_layers": 2,
        "n_backbone_size": 8,
        "hidden_size": 8  
    })
    # configs.append({
    #     "name": "v1_test",  
    #     "n_backbone_layers": 3,
    #     "n_backbone_size": 72,
    #     "hidden_size": 80  
    # })

    # configs.append({
    #     "name": "v0_xs48",                
    #     "n_backbone_layers": 3,
    #     "n_backbone_size": 48,
    #     "hidden_size": 48
    # })
    # configs.append({
    #     "name": "v0_xs40",                
    #     "n_backbone_layers": 3,
    #     "n_backbone_size": 40,
    #     "hidden_size": 40
    # })
    # configs.append({
    #     "name": "v0_xs32",                
    #     "n_backbone_layers": 3,
    #     "n_backbone_size": 32,
    #     "hidden_size": 32
    # })
    # configs.append({
    #     "name": "v1_smallest",                
    #     "n_backbone_layers": 3,
    #     "n_backbone_size": 56,
    #     "hidden_size": 56
    # })
    # configs.append({
    #     "name": "v2_besttuned",                
    #     "n_backbone_layers": 3,
    #     "n_backbone_size": 72,
    #     "hidden_size": 80  
    # })
    # configs.append({
    #     "name": "v3_large",                
    #     "n_backbone_layers": 4,
    #     "n_backbone_size": 72,
    #     "hidden_size": 72  
    # })
    # configs.append({
    #     "name": "v4_larger",        
    #     "n_backbone_layers": 4,
    #     "n_backbone_size": 72,
    #     "hidden_size": 72  
    # })
    # configs.append({
    #     "name": "v5_largest",        
    #     "n_backbone_layers": 4,
    #     "n_backbone_size": 80,
    #     "hidden_size": 80  
    # })

    try:
        for config in configs:
            config.update(default_config)
            train(config)
    finally:
        ray.shutdown()


def resume_training():
    ray.init(ignore_reinit_error=True, object_store_memory=300*1024*1024*1024)

    model_name = "v1_smallest"
    trainer_path = f'/root/scikit-map/multione/final/ray_results/{model_name}/TorchTrainer_2bd66_00000_0_2025-10-18_21-05-27'
    checkpoint = Path(trainer_path) / "checkpoint_000099"

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
        "name": "v1_smallest_continued",   
        "checkpoint_path": checkpoint.as_posix(),
        "lr": 0.0001,
        "batch_size": 2048,      
        "max_num_epochs": 100,
        "dataset": dsr,
        "name": "v1_smallest",                
        "n_backbone_layers": 3,
        "n_backbone_size": 56,
        "hidden_size": 56
    }

    try:
        
        train(default_config)
    finally:
        ray.shutdown()


def monitor_training():
    model_name = "v0_xs48"
    fld_result = list((fld_ray_results / model_name).glob('TorchTrainer_*'))[-1]

    results = ray.train.Result.from_path(fld_result)
    df = results.metrics_dataframe
    print(df)
    
if __name__ == "__main__":
    main()
    #resume_training()
