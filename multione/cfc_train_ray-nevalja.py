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
FN_ZARR = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
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
    batchsize = config.get("batch_size", BATCHSIZE)
    ds: ArcoV2Dataset = ray.get(config["dataset"])
    train_subset, val_subset = ds.get_train_validation_subset(ncases_validation=0.2)

    train_loader = MemoryDataLoader(ds, batch_size=batchsize, indexes=train_subset.indices, cache_length=100)
    val_loader = MemoryDataLoader(ds, batch_size=batchsize, indexes=val_subset.indices, cache_length=100)
    # ds_train: DataIterator = ray.train.get_dataset_shard("train")
    # ds_valid: DataIterator = ray.train.get_dataset_shard("valid")

    for epoch in range(EPOCHS):
        if ray.train.get_context().get_world_size() > 1:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        rmse = torchmetrics.MeanSquaredError(squared=False)
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
            rmse = torchmetrics.MeanSquaredError(squared=False)
            for (y, x, tl, ts) in val_loader:
                outputs = model(x, tl, ts)
                #loss = criterion(outputs, y)
                rmse.update(outputs, y)

        avg_val_loss = rmse.compute().item()

        # Log metrics
        ray.train.report({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch})


#%%
def train(config, dataset):
    

    config = {
        "lr": 1e-4,        
        "n_backbone_layers": 4,
        "n_backbone_size": 64,
        "hidden_size": 64,
    }

    run_config = RunConfig(storage_path="/root/arcov2/trainray", name="run_0")
    scaling_config = ScalingConfig(num_workers=8, use_gpu=True, resources_per_worker={"CPU":2, "GPU": 0.5})
    #ds_train, ds_valid = ray.data.from_torch(dataset_torch).train_test_split(test_size=0.2, shuffle=True, seed=47)

    # train_subset, val_subset = dataset.get_train_validation_subset(ncases_validation=0.2)

    # train_loader = MemoryDataLoader(dataset, batch_size=BATCHSIZE, indexes=train_subset.indices, cache_length=100)
    # val_loader = MemoryDataLoader(dataset, batch_size=BATCHSIZE, indexes=val_subset.indices, cache_length=100)

    trainer = TorchTrainer(train_func, 
                        scaling_config=scaling_config, 
                        run_config=run_config,
                        datasets={"train": train_loader, "valid": val_loader})
    result = trainer.fit()

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
        return_tensors=False
    )
    dataset.prepare_all_cases()

    dsr = ray.put(dataset)

    # ds_y = ray.data.from_numpy(dataset.all_y, override_num_blocks = 16)
    # ds_x = ray.data.from_numpy(dataset.all_x)
    # ds_x_gtemp = ray.data.from_numpy(dataset.all_x_gtemp)
    # ds_timespans = ray.data.from_numpy(dataset.all_timespans)    

    # ds = ray.data.from_items([dict(y=d[0], x=d[1], tl=d[2], ts=d[3]) for d in dataset]) # sporo


    #ds = ray.data.from_numpy()
    #ds = ray.data.from_items([dict(y=dataset.all_y, x=dataset.all_x, tl=dataset.all_x_gtemp, ts=dataset.all_timespans)]) # sporo
    # print(ds.schema())

    

    train({}, ds)