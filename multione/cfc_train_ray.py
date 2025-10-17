#%%
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
from ray.train import ScalingConfig, RunConfig

from cfc import CfcModel, ArcoV2Dataset, MemoryDataLoader

INDICES = ['ndvi']
OUTPUT_SIZE = len(INDICES)
INPUT_SIZE = len(INDICES) + 2
TIMELESS_SIZE = 3
SEQUENCE_LENGTH = 12
YEARS = range(2000, 2024)
FN_ZARR = "mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
LIMIT = 500
PERCENT_PIXELS = 0.1
BATCHSIZE = 1024
EPOCHS = 100
criterion = nn.MSELoss()

#%%
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
    optimizer = Adam(model.parameters(), lr=config["lr"])

    # Data
    ds_train: DataIterator = ray.train.get_dataset_shard("train")
    ds_valid: DataIterator = ray.train.get_dataset_shard("valid")

    for epoch in range(EPOCHS):
        model.train()   
        rmse = torchmetrics.MeanSquaredError(squared=True)     
        for (y, x, tl, ts) in ds_train.iter_torch_batches(batch_size=config["batch_size"], prefetch_batches=10):
            optimizer.zero_grad()
            outputs = model(x, tl, ts)
            rmse.update(outputs.detach(), y)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        avg_train_loss = rmse.compute().item()

        model.eval()
        with torch.no_grad():
            rmse = torchmetrics.MeanSquaredError(squared=True)
            for (y, x, tl, ts) in ds_valid.iter_torch_batches(batch_size=config["batch_size"], prefetch_batches=10):
                outputs = model(x, tl, ts)
                loss = criterion(outputs, y)
                rmse.update(outputs, y)

        avg_val_loss = rmse.compute().item()

        # Log metrics
        ray.train.report({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch})


#%%
def train(config, ds):
    

    config = {
        "lr": 1e-4,        
        "n_backbone_layers": 4,
        "n_backbone_size": 64,
        "hidden_size": 64,
    }

    run_config = RunConfig(storage_path="/root/arcov2/trainray", name="run_0")
    scaling_config = ScalingConfig(num_workers=4, use_gpu=True, resources_per_worker={"CPU":2, "GPU": 0.5})
    #ds_train, ds_valid = ray.data.from_torch(dataset_torch).train_test_split(test_size=0.2, shuffle=True, seed=47)




    ds_train, ds_valid = ds.train_test_split(test_size=0.2, shuffle=True, seed=47)

    trainer = TorchTrainer(train_func, 
                        scaling_config=scaling_config, 
                        run_config=run_config,
                        datasets={"train": ds_train, "valid": ds_valid})
    result = trainer.fit()

def main():
    # LIMIT=10; PERCENT_PIXELS = 0.3
    dataset = ArcoV2Dataset(
        zarr_path=FN_ZARR,
        years=YEARS,
        indices=INDICES,
        limit=LIMIT,
        percent_pixels=PERCENT_PIXELS,
        sequence_length=SEQUENCE_LENGTH,
    )

    ds = ray.data.from_torch(dataset)

    train({}, ds)