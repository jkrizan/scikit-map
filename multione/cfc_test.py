#%%
import gc
from numba import njit, prange

import rasterio
import torch
import ray, ray.train

from cfc import CfcModel, ArcoV2Dataset, MemoryDataLoader
import cfc_sample

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
from typing import Any
import multiprocessing as mp

# fld = Path(__file__).parent / "final"  #Hydra
fld = Path('/mnt/nibble/gen_cog/arcov2/final') 
fld_ray_results = (fld / "ray_results_xxs")

FN_ZARR = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
YEARS = range(2000, 2024)
LIMIT =  slice(500, None)
PERCENT_PIXELS = 0.01
INDICES = ['fpar']
SEQUENCE_LENGTH = 12
TIMELESS_SIZE = 3

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

def create_model(config, fld_checkpoint: Path) -> CfcModel:
    OUTPUT_SIZE = len(INDICES)
    INPUT_SIZE = len(INDICES) + 2
    
    SEQUENCE_LENGTH = 12

    model = CfcModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        timeless_input_size=TIMELESS_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        backbone_layers = [config["n_backbone_size"]] * config["n_backbone_layers"],        
        hidden_size=config["hidden_size"],
    )
    
    ray_state_dict = torch.load(fld_checkpoint/"checkpoint.pt", map_location='cpu')[0]
    state_dict = {k.replace('module.', ''): v for k, v in ray_state_dict.items()}
    model.load_state_dict(state_dict)

    return model
    
def load_dataset(percent_pixels: float|None, limit: slice | None) -> ArcoV2Dataset:
    if percent_pixels is None:
        percent_pixels = PERCENT_PIXELS
    if limit is None:
        limit = LIMIT
    ds = ArcoV2Dataset(
        zarr_path=FN_ZARR,
        years=YEARS,
        indices=INDICES,
        sequence_length=SEQUENCE_LENGTH,
        percent_pixels=percent_pixels,
        limit=limit,
        device='cpu'
    )
    return ds

def find_best_epoch(model_name):
    fld_trainer = sorted((fld_ray_results / model_name).glob("TorchTrainer*"))[-1]
    #results = ray.train.Result.from_path(fld_trainer)
    #df: pandas.DataFrame = results.metrics_dataframe    # type: ignore
    df = pandas.read_csv(fld_trainer / "progress.csv")
    best_epoch = df.sort_values('val_loss').iloc[0]

    # Model configuration
    config = json.load(open(fld_trainer / "params.json", "r"))['train_loop_config']   

    # Load best model and print parameters
    fld_checkpoint = fld_trainer / best_epoch.checkpoint_dir_name
    model = create_model(config, fld_checkpoint)

    return best_epoch, df, model
#%%
def train_stats(model_name: str, ds, ds_prepared, subname = None):
    # model_name = 'v1_smallest'; subname = 'e198'
    # model_name = 'v0_xs_2_8_8'; subname = None
    fld_out = fld / model_name
    fld_out.mkdir(exist_ok=True, parents=True)

    with Logger(fld_out / f"{model_name}_test_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log") as log:
        # log = Logger(fld_out / f"{model_name}_test_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        # Training results

        best_epoch, df, model = find_best_epoch(model_name)

        log(f"Model: {model_name}")
        log(f"Best epoch: {best_epoch['epoch']}, val_loss: {best_epoch['val_loss']}, train_loss: {best_epoch['train_loss']}")

        # Plot training and validation loss
        val_loss: np.ndarray = df.sort_values('epoch')['val_loss'].values   # type: ignore
        train_loss: np.ndarray = df.sort_values('epoch')['train_loss'].values   # type: ignore
        plt.figure(figsize=(10,5))
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        plt.ylim(train_loss.min() - (val_loss[5] -val_loss.min())*0.2, val_loss[0]*1.01)
        plt.xlabel('Epoch')
        plt.ylabel('RMSE Loss')
        plt.title(f'Training and Validation Loss for {model_name}')
        plt.legend()
        plt.grid()
        plt.savefig(fld_out / f"{model_name}_{subname}-training_validation_loss.png", dpi=200)
        plt.show()

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"Model: {model_name}")
        log(f"Total parameters: {total_params}")
        log(f"Trainable parameters: {trainable_params}")

        # Load test dataset, cases that were not used in training/validation
        # PERCENT_PIXELS = 0.1; LIMIT = slice(500, 520)
        #ds = load_dataset(percent_pixels=PERCENT_PIXELS, limit=LIMIT)
        #n_samples = len(ds)
        #log(f"Test Dataset loaded with {n_samples} samples.")

        # Timeseries drawing
        fld_timeseries = fld_out / "timeseries"
        fld_timeseries.mkdir(exist_ok=True, parents=True)
        gen = np.random.default_rng(seed=42)
        for i in range(10):
            tile_ind = gen.integers(0, len(ds.tiles))
            pix_ind = gen.integers(0, ds.data[tile_ind][-1].shape[0])
            fig = draw_timeseries(ds, model, tile_ind, pix_ind)
            fig.savefig(fld_timeseries / f"{model_name}{'' if subname is None else f'_{subname}'}_timeseries_tile{tile_ind}_pix{pix_ind}.png", dpi=200)
            plt.close(fig)


        # log("Preparing all cases...")
        # ds.prepare_all_cases()
        # log("All cases prepared.")

        # Loop over optimization techniques? 
        # model = optimize_model(model, ds)

        # Evaluate on test dataset
        # optimizations = [("Original", lambda m,d: m, '_orig'),
        #                  ("Torch compile", optimize_model_torch, '_torch'),
        #                  ("OpenVINO onnx", optimize_model_openvino, '_openvino'),
        #                  #("Torch comile FP16", optimize_model_fp16, '_fp16'),  # Very slow!!!
        #                  #("Torch comile BFP16", optimize_model_fp16, '_bfp16'), # BFloat16 - not accurate!
        #                  ]
    
        # Depends on batch_size!
        # 5E3 =1.7min, cpu_count()*10= 6.9min, cpu_count()*100=1.2min, cpu_count()*1E3=1.1min, cpu_count()*1E4=1.9min
        #  cpu_count()*2*1E3 = 1.4min, cpu_count()*500=1min, cpu_count()*300=0.9min, cpu_count()*200=1min
        # opt_model = optimize_model_bfp16(model, ds)   # 1.1min, first pass is slow
        model = model.to(memory_format=torch.channels_last)
        #%%
        # opt_model = optimize_model_torch(model, ds_prepared)  # 1.0min
        opt_model = optimize_model_amp(model, enable_amp=True)  # 1.0min
        #opt_model = optimize_model_bfp16(model, ds_prepared)   # 1.1min, first pass is slow
        opt_model = optimize_model_openvino(model, ds_prepared)  # This is best optimized model for CPU inference
        dl = MemoryDataLoader(ds_prepared, batch_size=mp.cpu_count()*300, indexes = None, shuffle=False) #mp.cpu_count()*300
        # float16 works only on GPU, 
        # didn't test quantization and bfloat16 
        prdy_fw=[]; prdy_bw=[]
        with torch.inference_mode():
            time0 = time.time()
            for i, (yb, xb, tlb, tsb) in tqdm.tqdm(enumerate(dl), total=len(dl)):                
                # (yb, xb, tlb, tsb) = next(iter(dl))
                #prd = model.inference(xb, tlb, tsb, direction='both').detach().squeeze()
                #xb = xb.to(torch.bfloat16); tlb = tlb.to(torch.bfloat16); tsb = tsb.to(torch.bfloat16)
                prd = opt_model(xb, tlb, tsb).detach().squeeze()
                prdy_fw.append(prd[:,0].squeeze())
                prdy_bw.append(prd[:,1].squeeze())
            time1 = time.time()
        samples_per_second = n_samples / (time1 - time0)
        log(f"Processed {n_samples} samples in {time1-time0:.1f} seconds, {samples_per_second:.1f} samples/second")
        log(f"Average time per one month: {4004*4004/samples_per_second/60:.1f} minutes")

        #%%

        prdy_fw = torch.cat(prdy_fw, dim=0)
        prdy_bw = torch.cat(prdy_bw, dim=0)

        res_fw = ds_prepared.all_y[:,:,0].squeeze() - prdy_fw
        res_bw = ds_prepared.all_y[:,:,1].squeeze() - prdy_bw

        mae_fw = torch.abs(res_fw).mean(dim=0)
        mse_fw = (res_fw ** 2).mean(dim=0)
        rmse_fw = torch.sqrt(mse_fw)
        r2_fw = torcheval.metrics.functional.r2_score(prdy_fw, ds_prepared.all_y[:,:,0].squeeze(), multioutput='raw_values')
        log(f"Forward predictions - MAE: {mae_fw: 0.4f}, MSE: {mse_fw: 0.4f}, RMSE: {rmse_fw: 0.4f}, R2: {r2_fw: 0.4f}")

        mae_bw = torch.abs(res_bw).mean(dim=0)
        mse_bw = (res_bw ** 2).mean(dim=0)
        rmse_bw = torch.sqrt(mse_bw)
        r2_bw = torcheval.metrics.functional.r2_score(prdy_bw, ds_prepared.all_y[:,:,1].squeeze(), multioutput='raw_values')
        log(f"Backward predictions - MAE: {mae_bw: 0.4f}, MSE: {mse_bw: 0.4f}, RMSE: {rmse_bw: 0.4f}, R2: {r2_bw: 0.4f}")

    #%%
# prepare all data for all pixels                        
@njit(parallel=True, fastmath=True, cache=True)
def _process_one_image_fpar(day_from_start, n_pixels,  
                    x, timeless, timespans, valid_pixels_ind,
                    days_from_start, 
                    landsat_data, 
                    modis_data,
                    geom_temp_doy,
                    valid_values_mask,
                    sequence_length):
    
    n_dates = days_from_start.shape[0]   
                                
    nir_ind = 1; red_ind = 0
    scale = (0.95 - 0.001)/(0.96 - 0.03)
    for pix in prange(n_pixels):
        valid_values = valid_values_mask[:,pix]           
        pix_dfs = days_from_start[valid_values]
        ind = np.nonzero(pix_dfs < day_from_start)[0][-sequence_length:]
        dfs = pix_dfs[ind]
        if len(ind) < sequence_length:
            valid_pixels_ind[pix] = False
            continue
        #for b in bands:
        #    x[pix, :, b] = landsat_data[b*n_dates:(b+1)*n_dates, pix][valid_values][ind]
        nir = landsat_data[nir_ind*n_dates:(nir_ind+1)*n_dates, pix][valid_values][ind]
        red = landsat_data[red_ind*n_dates:(red_ind+1)*n_dates, pix][valid_values][ind]
        denom = nir + red + 1e-6
        #denom[denom == 0] = np.nan  # to avoid division by zero
        ndvi = (nir - red) / denom
        
        x[pix,:,0] = np.clip((ndvi - 0.03) * scale + 0.001, 0.0, 1.0)  # fpar
        x[pix, :, 1] = modis_data[valid_values, pix][ind]
        ind_doys = dfs % 23
        x[pix, :, 2] = geom_temp_doy[ind_doys, pix]
        timespans[pix, 0] = 16 # TODO: FIX for backward
        timespans[pix, 1:-1] = dfs[1:] - dfs[:-1]
        timespans[pix, -1] = day_from_start - dfs[-1]
                                        
        # gtemp at prediction date
        last_dfs = dfs[-1]
        last_gtemp = geom_temp_doy[ind_doys[-1], pix]
        next_dfs_mask = days_from_start > day_from_start
        if np.any(next_dfs_mask):
            next_dfs = days_from_start[next_dfs_mask][0]
            next_gtemp = geom_temp_doy[next_dfs % 23, pix]                                                                        
            timeless[pix, 2] = last_gtemp + (next_gtemp - last_gtemp) * (day_from_start - last_dfs) / (next_dfs - last_dfs)  # gtemp at prediction date
        else:
            timeless[pix, 2] = last_gtemp
    return
#%%        
def predict_tiles(model_name, tiles = ['015E_43N','090W_49N', '055W_06S']):
    # model_name = 'v1_smallest'
    import utils
    from collections import namedtuple

    YearMonth = namedtuple('YearMonth', ['year', 'month'])
    dates = [YearMonth(year, month) for year in range(2023,2003,-1) for month in range(1, 13)]

    fld_predictions = fld / f"prediction_{model_name}"
    fld_predictions.mkdir(exist_ok=True, parents=True)
    ds = load_dataset(percent_pixels=0.1, limit=slice(500, 520))
    ds.prepare_all_cases()

    best_epoch, df, model = find_best_epoch(model_name)
    model = optimize_model_openvino(model, ds)

    profile=dict(
                driver='GTiff',
                count=1,
                dtype='uint16',
                width=utils.x_size,
                height=utils.y_size,                
                nodata=65535,
                blockxsize=1024, 
                blockysize=1024,    
                tiled=True,
                compress='deflate',
                predictor=2
            )
    nodata = profile['nodata']

    log = Logger(fld_predictions / f"{model_name}_tile_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log(f"Model: {model_name}")
    time0 = time.time()
    for tile in tiles:
        # tile = '015E_43N'
        time1 = time.time()
        gc.collect()
        log(f"Predicting tile {tile}...")

        (success, error, eta), meta, valid_data, data = cfc_sample.get_tile_data(tile)    
        (n_valid_pixels, inds_valid_pixels, valid_values_mask) = valid_data
        (landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy) = data

        duration = (time.time()-time1)
        log(f'Tile {tile} loaded in {duration/60:.2f} minutes')

        profile['crs'] = meta[0]
        profile['transform'] = meta[1]

        time1 = time.time()

        for yearmonth in dates:
            gc.collect()
            # yearmonth = dates[0]
            time2 = time.time()

            year, month = yearmonth
            log(f"Preparing data for {year}-{month:02d}")

            date = datetime(year, month, 15)
            day_from_start = (date - datetime(YEARS[0], 1, 1)).days - 1
            n_pixels = landsat_data.shape[1]        
            # n_dates = ds.days_from_start.shape[0]

            fns = list(fld_predictions.glob(f"{tile}_{year}{month:02d}_*.tif"))
            if len(fns) > 0:
                log(f"Predictions for {tile} {year}-{month:02d} already exist, skipping")
                continue
                
            timespans = np.empty((n_pixels, SEQUENCE_LENGTH + 1), dtype=np.float32)
            x = np.empty((n_pixels, SEQUENCE_LENGTH, len(INDICES) + 2), dtype=np.float32)
            timeless = np.empty((n_pixels, TIMELESS_SIZE), dtype=np.float32)
            valid_pixels_ind = np.ones(n_pixels, dtype=bool)

            _process_one_image_fpar(day_from_start, n_pixels,
                        x, timeless, timespans, valid_pixels_ind,
                        ds.days_from_start.numpy(),
                        landsat_data, modis_data, geom_temp_doy,
                        valid_values_mask,
                        SEQUENCE_LENGTH)

            timeless[:,0] = geom_temp_doy.min(axis=0)   # gtemp_min
            timeless[:,1] = geom_temp_doy.max(axis=0)   # gtemp_max

            if not valid_pixels_ind.all():
                x = x[valid_pixels_ind, :, :]
                timeless = timeless[valid_pixels_ind, :]
                timespans = timespans[valid_pixels_ind, :]
            timespans /= 366.0

            # x = torch.tensor(x, dtype=torch.bfloat16)
            # timeless = torch.tensor(timeless, dtype=torch.bfloat16)
            # timespans = torch.tensor(timespans, dtype=torch.bfloat16)

            duration = (time.time()-time2)
            log(f"Data for {year}-{month:02d} prepared in {duration/60:.2f} minutes, running prediction for {x.shape[0]} pixels")
            
            batch_size = mp.cpu_count()*300
            prd_list = []
            time3 = time.time()
            with torch.inference_mode():
                for i in range(0, x.shape[0], batch_size):
                    xb = x[i:i+batch_size, :, :]
                    tlb = timeless[i:i+batch_size, :]
                    tsb = timespans[i:i+batch_size, :]
                    prd_batch = model(xb, tlb, tsb).detach().squeeze().cpu().numpy()
                    prd_list.append(prd_batch)
                prd = np.concatenate(prd_list, axis=0)
                prd = prd[:,0]  # take forward direction
                prd = (np.clip(prd, 0, 1) * 40000).astype(np.uint16)

            duration = (time.time()-time3)            
            log(f"Prediction for {year}-{month:02d} done in {duration/60:.2f} minutes, saving to disk")

            time3 = time.time()
            if not valid_pixels_ind.all():
                img = np.full(n_pixels, nodata, dtype=np.uint16)
                img[valid_pixels_ind] = prd
                img = img.reshape((utils.y_size, utils.x_size))
            else:
                img = prd.reshape((utils.y_size, utils.x_size))

            fn = fld_predictions / f"{tile}_{year}{month:02d}_fpar.tif"

            with rasterio.open(fn, 'w', **profile) as dst:
                dst._set_all_scales([1./40000.])
                dst._set_all_offsets([0.0])
                dst.write(img, 1)
            duration = (time.time()-time3)
            log(f"Prediction for {year}-{month:02d} done in {duration/60:.2f} minutes, saving to {fn}")

        duration = (time.time()-time1)
        log(f"Total time for tile {tile}: {duration/60:.2f} minutes")                
    log(f"Total time elapsed: {(time.time()-time0)/60:.2f} minutes")

def draw_timeseries(ds: ArcoV2Dataset, model: CfcModel, tile_ind: int, pix_ind: int):
        # tile_ind=3; pix_ind=100        

        (directions, y, x, timeless, timespans, dates, valid_values) = ds.get_one_pixel_timeseries(tile_ind, pix_ind)
        #inds = self.dataset.get_one_pixel_indices(tile_ind, pix_ind)
        # inds = [1202496, 1202497, 1202498, 1202499, 1202500, 1202501, 1202502]
        #(y, x, timeless, timespans) = self.dataset.get_cases(inds)
        # x = x.permute(0,2,1)  # (n_dates, seq_len, n_features)
        y_prd = model(x, timeless, timespans).detach().numpy()
        y_prd = np.take_along_axis(y_prd, directions[np.newaxis,np.newaxis,:].T,axis=2).squeeze()
        y = y.numpy()

        yv = y[valid_values].squeeze(); 
        yv_prd = y_prd[valid_values]
        resv = yv - yv_prd
        rmse = np.sqrt(np.mean(resv**2))
        mae = (np.abs(resv)).mean()
        r2 = 1-np.var(resv) / np.var(yv)
        #std = resv.std()
        vdates = dates[valid_values]

        band_name = INDICES[0].split('_')[0].upper()

        fig, ax = plt.subplots(1,1,figsize=(12,5))            
        label = f"{band_name}\n$mae={mae:.4f}$\n$rmse={rmse:.4f}$\n$r^2={r2:.3f}$"
        ax.plot(vdates, yv, 'o', color='red', markersize=4, alpha=0.5)
        ax.plot(dates, y_prd, '-', label=label, color='blue')
        ax.legend()

        ax.set_title(f"Tile {ds.tiles[tile_ind]}, Pixel {pix_ind}") #\nMAE: {mae[0]:.4f}, RMSE: {rmse[0]:.4f}, R2: {r2[0]:.4f}")
        ax.set_xlabel("Date")
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
        pixel_inds = group['pixel_inds'][:]  #type: ignore
        water_mask: np.ndarray = group['water_mask'][:].flatten()[pixel_inds] #type: ignore
        data.append(water_mask.flatten())

    data = np.concatenate(data, axis=0)
    data = data[data<=100]   # remove nodata values
    print(f"Water mask stats: min={data.min()}, max={data.max()}, mean={data.mean()}, std={data.std()}")
    n_less_95 = np.sum(data < 95)
    p_less_95 = n_less_95 / len(data) * 100
    print(f"Percentage of pixels with water percentage < 95%: {p_less_95:.2f}%")

    plt.figure(figsize=(10,5))
    #sns.violinplot(x=data, inner="quart")    
    plt.hist(data, bins=100, range=(0,100), density=True)
    plt.title(f"Water Mask Histogram\n{p_less_95:.2f}% of pixels with water percentage < 95%")
    plt.axvline(95, color='red', linestyle='--')
    plt.yscale('log')
    plt.xlabel("Water percentage")
    plt.ylabel("Percentage of pixels (log scale)")
    plt.grid()
    plt.savefig(fld / "testdata_water_mask_histogram.png", dpi=200)
    plt.show()

def optimize_model_amp(model, enable_amp=True):
    def infer_fn(xb, tlb, tsb):
        with torch.inference_mode(), torch.amp.autocast(
                'cpu',
                dtype=torch.bfloat16,
                enabled=enable_amp
        ):
            output = model(xb, tlb, tsb)
        return output
    return infer_fn

def optimize_model_torch(model: CfcModel, dataset: ArcoV2Dataset) -> CfcModel:
    # import torch.quantization as quant
    model = model.eval()
    model.compile(fullgraph = True)
    return model
    
def optimize_model_bfp16(model: CfcModel, dataset: ArcoV2Dataset) -> CfcModel:
    model = model.eval()
    model.bfloat16()
    dataset.all_x = dataset.all_x.to(torch.bfloat16)
    dataset.all_x_gtemp = dataset.all_x_gtemp.to(torch.bfloat16)
    dataset.all_timespans = dataset.all_timespans.to(torch.bfloat16)
    #model = optimize_model_openvino(model, dataset)
    model.compile(fullgraph = True)
    return model
    
def optimize_model_openvino(model: CfcModel, dataset: ArcoV2Dataset) -> Any:
    from openvino import Core, properties
    import torch.onnx

    # model = create_model(config, fld_checkpoint)
    model = model.eval()
    # Export the model to ONNX

    _, dummy_input_x, dummy_input_tl, dummy_input_ts = dataset.get_cases(np.random.choice(len(dataset), size=1024, replace=False))
    onnx_model_path = "temp_model.onnx"
    torch.onnx.export(model, (dummy_input_x, dummy_input_tl, dummy_input_ts), onnx_model_path,
                      input_names=['input_x', 'input_tl', 'input_ts'],
                      output_names=['output'],
                      dynamic_axes={'input_x': {0: 'batch_size'},
                                    'input_tl': {0: 'batch_size'},
                                    'input_ts': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}})

    # Load the ONNX model with OpenVINO
    core = Core()
    ov_model = core.read_model(model=onnx_model_path)
    compile_config = {  # Best practices for CPU performance in comments
        properties.inference_num_threads(): mp.cpu_count()//2,  # mp.cpu_count()//2
        properties.hint.enable_hyper_threading(): False,    # False
        properties.hint.enable_cpu_pinning(): True, # True
        properties.hint.performance_mode(): properties.hint.PerformanceMode.LATENCY, #LATENCY
        # the value of ov::num_streams is calculated by dividing ov::inference_num_threads by the number of threads per stream.
        # properties.num_streams(): mp.cpu_count() // 2,  # assuming 4 threads per stream
        }    
    compiled_model = core.compile_model(ov_model, device_name="CPU", config=compile_config)

    class OpenVINOModelWrapper(torch.nn.Module):
        def __init__(self, compiled_model):
            super(OpenVINOModelWrapper, self).__init__()
            self.compiled_model = compiled_model

        def forward(self, x, tl, ts):
            inputs = {
                'input_x': x, #.numpy(),
                'input_tl': tl, #.numpy(),
                'input_ts': ts #.numpy()
            }
            result = self.compiled_model.infer_new_request(inputs)
            return torch.from_numpy(result['output'])

    model = OpenVINOModelWrapper(compiled_model)
    return model
#%%

def test_filelock():
    from filelock import FileLock, Timeout
    from pathlib import Path

    lock_path = "test.lock"
    lock = FileLock(lock_path, timeout=1)
    try:
        with lock:
            print("Lock acquired.")            
            time.sleep(5)
    except Timeout:
        print("Could not acquire lock.")

def test_optimizatons():
    model_name = 'v0_xs_2_8_8'
    ds = load_dataset(percent_pixels=PERCENT_PIXELS, limit=LIMIT)
    ds.prepare_all_cases()
    n_samples = len(ds)
    best_epoch, df, model = find_best_epoch(model_name)

    # model_opt_torch = optimize_model_torch(model, ds)
    # model_opt_bfp16 = optimize_model_bfp16(model, ds)
    # model_opt_openvino = optimize_model_openvino(model, ds)

    opt_model = optimize_model_openvino(model, ds)  # This is best optimized model for CPU inference
    dl = MemoryDataLoader(ds, batch_size=mp.cpu_count()*300, indexes = None, shuffle=False) #mp.cpu_count()*300
    # float16 works only on GPU, 
    # didn't test quantization and bfloat16 
    prdy_fw=[]; prdy_bw=[]
    with torch.inference_mode():
        time0 = time.time()
        for i, (yb, xb, tlb, tsb) in tqdm.tqdm(enumerate(dl), total=len(dl)):                
            # (yb, xb, tlb, tsb) = next(iter(dl))
            #prd = model.inference(xb, tlb, tsb, direction='both').detach().squeeze()
            #xb = xb.to(torch.bfloat16); tlb = tlb.to(torch.bfloat16); tsb = tsb.to(torch.bfloat16)
            prd = opt_model(xb, tlb, tsb).detach().squeeze()
            prdy_fw.append(prd[:,0].squeeze())
            prdy_bw.append(prd[:,1].squeeze())
        time1 = time.time()
    samples_per_second = n_samples / (time1 - time0)
    print(f"Processed {n_samples} samples in {time1-time0:.1f} seconds, {samples_per_second:.1f} samples/second")
    print(f"Average time per one month: {4004*4004/samples_per_second/60:.1f} minutes")

#%%

#%%
if __name__ == "__main__":
    #water_mask_stats()
    #predict_tiles('v0_xs40', tiles = ['015E_43N','090W_49N', '055W_06S'])
    # model_names = [
    #     'v1_smallest',
    #     'v2_small',
    #     'v3_medium',
    #     'v4_large',
    #     'v5_xlarge',
    # ]

    # ds = load_dataset(percent_pixels=PERCENT_PIXELS, limit=LIMIT)
    # n_samples = len(ds)
    # print(f"Test Dataset loaded with {n_samples} samples.")
    # ds_prepared = load_dataset(percent_pixels=PERCENT_PIXELS, limit=LIMIT)
    # ds_prepared.prepare_all_cases()
    # models = [mdir.name for mdir in fld_ray_results.iterdir()] 
    # for model_name in models:
    #     train_stats(model_name, ds, ds_prepared)
    
    test_optimizatons()