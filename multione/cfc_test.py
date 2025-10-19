#%%
import torcheval.metrics
import torch
import ray, ray.train

from cfc import CfcModel, ArcoV2Dataset, MemoryDataLoader

from pathlib import Path
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import json
import time
import tqdm

fld = Path(__file__).parent / "final" 
fld_ray_results = (fld / "ray_results")

FN_ZARR = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
YEARS = range(2000, 2024)
LIMIT =  slice(500, None)
PERCENT_PIXELS = 1.0
INDICES = ['fpar']
SEQUENCE_LENGTH = 12

#%%
def create_model(config):
    INDICES = ['fpar']
    OUTPUT_SIZE = len(INDICES)
    INPUT_SIZE = len(INDICES) + 2
    TIMELESS_SIZE = 3
    SEQUENCE_LENGTH = 12

    model = CfcModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        timeless_input_size=TIMELESS_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        backbone_layers = [config["n_backbone_size"]] * config["n_backbone_layers"],        
        hidden_size=config["hidden_size"],
    )
    return model
    

def train_stats(model_name: str):
    # model_name = 'v1_smallest'
    fld_out = fld / model_name
    fld_out.mkdir(exist_ok=True, parents=True)

    # Training results
    fld_trainer = sorted((fld_ray_results / model_name).glob("TorchTrainer*"))[-1]
    results = ray.train.Result.from_path(fld_trainer)
    df: pandas.DataFrame = results.metrics_dataframe    # type: ignore
    best_epoch = df.sort_values('val_loss').iloc[0]
    print(f"Model: {model_name}")
    print(f"Best epoch: {best_epoch['epoch']}, val_loss: {best_epoch['val_loss']}, train_loss: {best_epoch['train_loss']}")

    # Plot training and validation loss
    val_loss: np.ndarray = df.sort_values('epoch')['val_loss'].values   # type: ignore
    plt.figure(figsize=(10,5))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
    plt.ylim(val_loss.min() - (val_loss[5] -val_loss.min())*0.2, val_loss[5])
    plt.xlabel('Epoch')
    plt.ylabel('RMSE Loss')
    plt.title(f'Training and Validation Loss for {model_name}')
    plt.legend()
    plt.grid()
    plt.savefig(fld_out / f"{model_name}-training_validation_loss.png", dpi=200)
    plt.show()

    # Model configuration
    config = json.load(open(fld_trainer / "params.json", "r"))['train_loop_config']
    model = create_model(config)
    
    # Load best model and print parameters
    fld_checkpoint = fld_trainer / best_epoch.checkpoint_dir_name
    ray_state_dict = torch.load(fld_checkpoint/"checkpoint.pt", map_location='cpu')[0]
    state_dict = {k.replace('module.', ''): v for k, v in ray_state_dict.items()}
    model.load_state_dict(state_dict)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # Load test dataset, cases that were not used in training/validation
    # PERCENT_PIXELS = 0.1; LIMIT = slice(500, 520)
    ds = ArcoV2Dataset(
        zarr_path=FN_ZARR,
        years=YEARS,
        indices=INDICES,
        limit=LIMIT,
        percent_pixels=PERCENT_PIXELS,
        sequence_length=SEQUENCE_LENGTH,
        return_tensors=True,
        device='cpu'
    )
    n_samples = len(ds)
    print(f"Test Dataset loaded with {n_samples} samples.")
    print("Preparing all cases...")
    ds.prepare_all_cases()
    print("All cases prepared.")

    # Loop over optimization techniques? 
    # model = optimize_model(model, ds)

    # Evaluate on test dataset
    dl = MemoryDataLoader(ds, batch_size=1024*10, indexes = None, shuffle=False)

    prdy_fw=[]; prdy_bw=[]
    time0 = time.time()
    for i, (yb, xb, tlb, tsb) in tqdm.tqdm(enumerate(dl), total=len(dl)):
        # (yb, xb, tlb, tsb) = next(iter(dl))
        prd = model.inference(xb, tlb, tsb, direction='both').detach().squeeze()
        prdy_fw.append(prd[:,0].squeeze())
        prdy_bw.append(prd[:,1].squeeze())
    time1 = time.time()
    samples_per_second = n_samples / (time1 - time0)
    print(f"Processed {n_samples} samples in {time1-time0:.1f} seconds, {samples_per_second:.1f} samples/second")
    print(f"Average time per one month: {4004*4004/samples_per_second/60:.1f} minutes")

    prdy_fw = torch.cat(prdy_fw, dim=0)
    prdy_bw = torch.cat(prdy_bw, dim=0)

    res_fw = ds.all_y[:,:,0].squeeze() - prdy_fw
    res_bw = ds.all_y[:,:,1].squeeze() - prdy_bw

    mae_fw = torch.abs(res_fw).mean(dim=0)
    mse_fw = (res_fw ** 2).mean(dim=0)
    rmse_fw = torch.sqrt(mse_fw)
    r2_fw = torcheval.metrics.functional.r2_score(prdy_fw, ds.all_y[:,:,0].squeeze(), multioutput='raw_values')
    print(f"Forward predictions - MAE: {mae_fw: 0.4f}, MSE: {mse_fw: 0.4f}, RMSE: {rmse_fw: 0.4f}, R2: {r2_fw: 0.4f}")

    mae_bw = torch.abs(res_bw).mean(dim=0)
    mse_bw = (res_bw ** 2).mean(dim=0)
    rmse_bw = torch.sqrt(mse_bw)
    r2_bw = torcheval.metrics.functional.r2_score(prdy_bw, ds.all_y[:,:,1].squeeze(), multioutput='raw_values')
    print(f"Backward predictions - MAE: {mae_bw: 0.4f}, MSE: {mse_bw: 0.4f}, RMSE: {rmse_bw: 0.4f}, R2: {r2_bw: 0.4f}")




#%%
def water_mask_stats():
    import zarr

    data=[]
    dataset: zarr.Group = zarr.open(FN_ZARR, mode='r') #type: ignore
    tiles = list(dataset.keys())
    for tile in tqdm.tqdm(tiles):
        group: zarr.Group = dataset[tile]   #type: ignore
        if 'water_mask' not in list(group.array_keys()):
            print(f"Tile {tile} has no water mask, skipping")
            continue
        water_mask: np.ndarray = group['water_mask'][:] #type: ignore
        data.append(water_mask.flatten())

    data = np.concatenate(data, axis=0)
    data = data[data<=100]   # remove nodata values
    print(f"Water mask stats: min={data.min()}, max={data.max()}, mean={data.mean()}, std={data.std()}")

    plt.figure(figsize=(10,5))
    sns.violinplot(x=data, inner="quart")    
    plt.title("Water Mask Histogram")
    plt.xlabel("Water percentage")
    plt.ylabel("Frequency")
    plt.grid()
    plt.savefig(fld / "testdata_water_mask_histogram.png", dpi=200)
    plt.show()

 def optimize_model(model: CfcModel, dataset: ArcoV2Dataset) -> CfcModel:
    import torch.quantization as quant
    model.eval()
    model.qconfig = quant.get_default_qconfig('fbgemm')
    quant.prepare(model, inplace=True)
    quant.convert(model, inplace=True)
    return model    
