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
from datetime import datetime

# fld = Path(__file__).parent / "final"  #Hydra
fld = Path('/mnt/nibble/gen_cog/arcov2/final') 
fld_ray_results = (fld / "ray_results")

FN_ZARR = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
YEARS = range(2000, 2024)
LIMIT =  slice(500, None)
PERCENT_PIXELS = 1.0
INDICES = ['fpar']
SEQUENCE_LENGTH = 12

#%%
class Logger():
    def __init__(self, fname):
        self.fname = fname
        self.file = open(self.fname, "wt")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __call__(self, msg: str) -> None:
        self.log(msg)

    def log(self, msg: str):
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()

    def close(self):
        self.file.close()

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

    with Logger(fld_out / f"{model_name}_test_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log") as log:
        # log = Logger(fld_out / f"{model_name}_test_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        # Training results
        fld_trainer = sorted((fld_ray_results / model_name).glob("TorchTrainer*"))[-1]
        results = ray.train.Result.from_path(fld_trainer)
        df: pandas.DataFrame = results.metrics_dataframe    # type: ignore
        best_epoch = df.sort_values('val_loss').iloc[0]
        log(f"Model: {model_name}")
        log(f"Best epoch: {best_epoch['epoch']}, val_loss: {best_epoch['val_loss']}, train_loss: {best_epoch['train_loss']}")

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
        log(f"Model: {model_name}")
        log(f"Total parameters: {total_params}")
        log(f"Trainable parameters: {trainable_params}")

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
        log(f"Test Dataset loaded with {n_samples} samples.")
        log("Preparing all cases...")
        ds.prepare_all_cases()
        log("All cases prepared.")

        # Loop over optimization techniques? 
        # model = optimize_model(model, ds)

        # Evaluate on test dataset
    #%%
        xb = ds.all_x[:10240].to(torch.bfloat16)   # (n_samples, seq_len, n_features)
        tlb = ds.all_x_gtemp[:10240].to(torch.bfloat16)  # (n_samples, timeless_size)
        tsb = ds.all_timespans[:10240].to(torch.bfloat16)

        optimizations = [("Original", lambda m,d: m, '_orig'),
                         ("Torch compile", optimize_model_torch, '_torch'),
                         #("Torch comile FP16", optimize_model_fp16, '_fp16'),  # Very slow!!!
                         #("Torch comile BFP16", optimize_model_fp16, '_bfp16'), # BFloat16 - not accurate!
                         ]
    #%%
        dl = MemoryDataLoader(ds, batch_size=1024*100, indexes = None, shuffle=False)
        prdy_fw=[]; prdy_bw=[]
        with torch.inference_mode():
            time0 = time.time()
            for i, (yb, xb, tlb, tsb) in tqdm.tqdm(enumerate(dl), total=len(dl)):                
                # (yb, xb, tlb, tsb) = next(iter(dl))
                #prd = model.inference(xb, tlb, tsb, direction='both').detach().squeeze()
                #xb = xb.to(torch.bfloat16); tlb = tlb.to(torch.bfloat16); tsb = tsb.to(torch.bfloat16)
                prd = model(xb, tlb, tsb).detach().squeeze()
                prdy_fw.append(prd[:,0].squeeze())
                prdy_bw.append(prd[:,1].squeeze())
            time1 = time.time()
        samples_per_second = n_samples / (time1 - time0)
        log(f"Processed {n_samples} samples in {time1-time0:.1f} seconds, {samples_per_second:.1f} samples/second")
        log(f"Average time per one month: {4004*4004/samples_per_second/60:.1f} minutes")

        prdy_fw = torch.cat(prdy_fw, dim=0)
        prdy_bw = torch.cat(prdy_bw, dim=0)

        res_fw = ds.all_y[:,:,0].squeeze() - prdy_fw
        res_bw = ds.all_y[:,:,1].squeeze() - prdy_bw

        mae_fw = torch.abs(res_fw).mean(dim=0)
        mse_fw = (res_fw ** 2).mean(dim=0)
        rmse_fw = torch.sqrt(mse_fw)
        r2_fw = torcheval.metrics.functional.r2_score(prdy_fw, ds.all_y[:,:,0].squeeze(), multioutput='raw_values')
        log(f"Forward predictions - MAE: {mae_fw: 0.4f}, MSE: {mse_fw: 0.4f}, RMSE: {rmse_fw: 0.4f}, R2: {r2_fw: 0.4f}")

        mae_bw = torch.abs(res_bw).mean(dim=0)
        mse_bw = (res_bw ** 2).mean(dim=0)
        rmse_bw = torch.sqrt(mse_bw)
        r2_bw = torcheval.metrics.functional.r2_score(prdy_bw, ds.all_y[:,:,1].squeeze(), multioutput='raw_values')
        log(f"Backward predictions - MAE: {mae_bw: 0.4f}, MSE: {mse_bw: 0.4f}, RMSE: {rmse_bw: 0.4f}, R2: {r2_bw: 0.4f}")
    #%%

def draw_timeseries(ds: ArcoV2Dataset, model: CfcModel, tile_ind: int, pix_ind: int):
        # tile_ind=3; pix_ind=100        

        (directions, y, x, timeless, timespans, dates, valid_values) = ds.get_one_pixel_timeseries(tile_ind, pix_ind)
        #inds = self.dataset.get_one_pixel_indices(tile_ind, pix_ind)
        # inds = [1202496, 1202497, 1202498, 1202499, 1202500, 1202501, 1202502]
        #(y, x, timeless, timespans) = self.dataset.get_cases(inds)
        # x = x.permute(0,2,1)  # (n_dates, seq_len, n_features)
        y_prd = model.forward(x, timeless, timespans).detach().numpy()
        y_prd = np.take_along_axis(y_prd,directions[np.newaxis,np.newaxis,:].T,axis=2).squeeze()
        y = y.numpy()

        yv = y[valid_values]; yv_prd = y_prd[valid_values]
        rmse = np.sqrt(np.mean((yv - yv_prd)**2, axis=0))
        mae = (np.abs(yv - yv_prd)).mean(axis=0)
        r2 = 1-np.var(yv - yv_prd, axis=0) / np.var(yv,axis=0)
        std = yv.std(axis=0)
        vdates = dates[valid_values]

        fig, ax = plt.subplots(6,1,figsize=(12,24), sharex=True)            
        for b in range(6):
            band_name = bands_prefix[b].split('_')[0].upper()
            label = f"{band_name}\n$rmse={rmse[b]:.4f}$\n$r^2={r2[b]:.3f}$\nstd={std[b]:.4f}$"
            ax[b].plot(vdates, yv[:,b], 'o', color='red', markersize=4, alpha=0.5)
            ax[b].plot(dates, y_prd[:,b], '-', label=label, color='blue')
            ax[b].legend()
        
        ax[0].set_title(f"Tile {self.dataset.tiles[tile_ind]}, Pixel {pix_ind}") #\nMAE: {mae[b]:.4f}, RMSE: {rmse[b]:.4f}, R2: {r2[b]:.4f}")
        ax[-1].set_xlabel("Date")
        fig.tight_layout()            

        return fig

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

def optimize_model_torch(model: CfcModel, dataset: ArcoV2Dataset) -> CfcModel:
    # import torch.quantization as quant
    model = model.eval()
    model.compile(fullgraph = True)
    return model
    
def optimize_model_fp16(model: CfcModel, dataset: ArcoV2Dataset) -> CfcModel:
    model = model.eval()
    model.bfloat16()
    model.compile(fullgraph = True)
    return model
    
    model.qconfig = quant.get_default_qconfig('fbgemm')
    quant.prepare(model, inplace=True)
    quant.convert(model, inplace=True)
    return model    
