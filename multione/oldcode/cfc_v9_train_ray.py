#%%
import os
import tempfile

from ray.data import DataIterator
from ray.data.datasource import SaveMode
from sqlalchemy.engine import row
import torch
import torch.nn  as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchmetrics
from torchvision.models import resnet18
from torchvision.datasets import FashionMNIST

import ray.train.torch
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig, RunConfig

from cfc_v9 import CfcLearnerV9, CfcModelV9, ArcoV2DatasetV9, MemoryDataLoader

INDICES = ['ndvi']
OUTPUT_SIZE = len(INDICES)
INPUT_SIZE = len(INDICES) + 2
TIMELESS_SIZE = 3
SEQUENCE_LENGTH = 12
YEARS = range(2000, 2024)
FN_ZARR = "/home/josip/arcov2/sample_v6.zarr"
LIMIT = None
PERCENT_PIXEL = 0.3
BATCHSIZE = 4096*2
EPOCHS = 100
criterion = nn.MSELoss()


def prepare_ray_dataset():
    dataset_torch = ArcoV2DatasetV9( 
                            FN_ZARR,                         
                            years=YEARS, 
                            sequence_length=SEQUENCE_LENGTH, 
                            indices=INDICES, 
                            limit=None, 
                            percent_pixels=PERCENT_PIXEL,
                            return_tensors=False)
    
    dataset_torch.prepare_all_cases()

    def transform_cases(row):
        y, x, tl, ts = row['item']
        # print(y.shape, x.shape, tl.shape, ts.shape)
        row['y'] = y
        row['x'] = x
        row['tl'] = tl
        row['ts'] = ts
        return row

    ray.init(ignore_reinit_error=True, object_store_memory=400*1024*1024*1024)
    ds = ray.data.from_torch(dataset_torch, local_read=True)
    #print(ds.schema())
    ds = ds.map(transform_cases, concurrency=16, memory=100*1024*1024*1024)
    ds = ds.drop_columns(['item'])
    #print(ds.schema())
    ds.write_parquet("/home/josip/arcov2/sample_v6.parquet", mode=SaveMode.OVERWRITE)

def test_dataset():
    ray.init(ignore_reinit_error=True, object_store_memory=400*1024*1024*1024)
    ds  = ray.data.read_parquet("/home/josip/arcov2/sample_v6.parquet")
    ds = ds.drop_columns(['item'])
    print(ds.schema())
    print(ds.count())
    print(ds.take(1))


#%%
def train_func(config):
    model = CfcModelV9(
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

ray.init(ignore_reinit_error=True, object_store_memory=400*1024*1024*1024)
prepare_ray_dataset()
exit()

config = {
    "lr": 1e-4,        
    "n_backbone_layers": 4,
    "n_backbone_size": 64,
    "hidden_size": 64,
}

run_config = RunConfig(storage_path="./cfc_v9_trainray", name="run_0")
scaling_config = ScalingConfig(num_workers=4, use_gpu=True, resources_per_worker={"CPU":2, "GPU": 0.5})
#ds_train, ds_valid = ray.data.from_torch(dataset_torch).train_test_split(test_size=0.2, shuffle=True, seed=47)




ds_train, ds_valid = ds.train_test_split(test_size=0.2, shuffle=True, seed=47)

trainer = TorchTrainer(train_func, 
                       scaling_config=scaling_config, 
                       run_config=run_config,
                       datasets={"train": ds_train, "valid": ds_valid})
result = trainer.fit()