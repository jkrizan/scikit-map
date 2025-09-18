#%%
#%%
from collections import namedtuple

from numpy.random import f
from settings import bands_prefix
import utils, cfc_sample
from utils import ttprint
from pathlib import Path
import pandas
import torcheval.metrics
from cfc_v7 import ArcoV2DatasetV7, CfcLearnerV7
import numpy as np  
import torch, torcheval
from torch.utils.data import DataLoader
from numba import njit, prange
import time, datetime
import tqdm
import pandas
import matplotlib.pyplot as plt
import rasterio 
import gc

fn_zarr = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
years = np.arange(2000, 2024)
sequence_length = 12

fld_checkpoints = "/mnt/nibble/gen_cog/arcov2/v6/checkpoints"
fld_tiffs = "/mnt/nibble/gen_cog/arcov2/v6/predictions"

YearMonth = namedtuple('YearMonth', ['year', 'month'])
#%%
class CfcV6Test:
    
    def __init__(self, 
                 fn_zarr: Path | str,   
                 fn_log: Path | str,              
                 device: str = 'cpu',
                 dtype: str = 'float32',
                 limit: int| None = None,
                 percent_pixel: float = 0.1,
                 ncases_validation: float = 0.1, 
                 ncases: int = int(10e6)
                ) -> None:
        
        self.fn_zarr = fn_zarr
        self.fn_log = Path(fn_log)
        self.limit = limit
        self.percent_pixels = percent_pixel
        self.ncases_validation = ncases_validation
        self.ncases = ncases

        if not self.fn_log.exists():
            self.fn_log.parent.mkdir(parents=True, exist_ok=True)
            self.fn_log.write_text("timestamp\tfn_checkpoint\tband\tmae\trmse\tr2\n")
        self.device = device
        if dtype=='float32':
            self.dtype = torch.float32
        elif dtype=='float16':
            self.dtype = torch.float16
        elif dtype=='bfloat16':
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.loaded_band=None

    def load_network(self, fn_checkpoint: Path | str):
        learner = CfcLearnerV6.load_from_checkpoint(fn_checkpoint)
        model = learner.model.to(self.device).to(self.dtype)
        #model.freeze()
        model.eval().compile(fullgraph=True)
        self.model = model

    def test_one_model(self, fn_checkpoint: Path, band: int                      ):
        # cfc_v6_b1_epoch-087.ckpt
        
        if self.loaded_band != band:
            self.load_dataset(band)

        self.load_network(fn_checkpoint)

        y=[]; prdy=[]
        time0 = time.time()
        for i, (yb, xb, tlb, tsb) in tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            # (yb, xb, tlb, tsb) = next(iter(dl))
            y.append(yb)
            prd = self.model(xb, tlb, tsb).detach()
            #prd = torch.tensor(model((xb.unsqueeze(1), tsb.unsqueeze(1)))[0])
            prdy.append(prd)
        runtime = time.time()-time0
        y = torch.cat(y, dim=0)
        prdy = torch.cat(prdy, dim=0).squeeze(1)
        print(f"Evaluation with done in {time.time()-time0:.1f} seconds")
        mae = (torch.abs(y - prdy)).mean(dim=0).item()
        mse = torcheval.metrics.functional.mean_squared_error(y, prdy).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()
        r2 = torcheval.metrics.functional.r2_score(y, prdy).item()
        ttprint(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")
        with self.fn_log.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{fn_checkpoint.name}\t{band}\t{mae}\t{rmse}\t{r2}\n")

    def load_dataset(self, band: int, prepare_all_cases: bool = True):
        self.dataset = ArcoV2DatasetV6(self.fn_zarr, 
                                       years, 
                                       sequence_length, 
                                       band=band,
                                       limit=self.limit,
                                       percent_pixels=self.percent_pixels, 
                                       device='cpu', 
                                       dtype=self.dtype)
        self.loaded_band = band
        ttprint(f"Dataset length: {len(self.dataset)}")

        if prepare_all_cases:
            self.dataset.prepare_all_cases()
        self.dataloader = DataLoader(self.dataset, batch_size=4096, shuffle=False, num_workers=8)

    def run_all(self, fld_checkpoints: Path | str):
        fld_checkpoints = Path(fld_checkpoints)
        fns_checkpoints = sorted(fld_checkpoints.glob("cfc_v6_*.ckpt"))
        fns = pandas.DataFrame([dict(fn=fn, band=int(fn.stem.split('_')[2][1:])) for fn in fns_checkpoints]) 
        bands = fns['band'].unique()
        ttprint(f"Found {len(fns_checkpoints)} checkpoints for bands: {bands}")

        df = pandas.read_csv(self.fn_log, sep="\t")
        fns_done = df['fn_checkpoint'].unique().tolist()
        fns = fns[~fns['fn'].apply(lambda x: x.name in fns_done)]
        ttprint(f"{len(fns_done)} checkpoints have been tested, {len(fns)} remaining")

        for b in bands:
            fns_band = fns[fns['band']==b]['fn'].tolist()
            if len(fns_band)==0:
                ttprint(f"All checkpoints for band {b} have been tested, skipping")
                continue
            ttprint(f"Loading dataset for band {b}")
            self.load_dataset(b)            
            for fn_checkpoint in fns_band:
                ttprint(f"Testing checkpoint: {fn_checkpoint}")
                self.test_one_model(fn_checkpoint, band=b)

    def draw_timeseries(self, models:list[str|Path]| str|Path,
                        n_random_pixels: int = 1,
                        ):
        # models = 'cfc_v6_b1_epoch-090.ckpt'
        if isinstance(models, (str, Path)):
            models = [Path(fld_checkpoints)/models]
        else:
            models = [Path(fld_checkpoints)/m for m in models]

        fns = pandas.DataFrame([dict(fn=fn, band=int(fn.stem.split('_')[2][1:])) for fn in models]) 

        for band in fns['band'].unique():
            # band=1
            band_name = bands_prefix[band].split('_')[0].upper()
            if self.loaded_band != band:
                self.load_dataset(band, prepare_all_cases=False)
            fns_band = fns[fns['band']==band]['fn'].tolist()
            for fn in fns_band:
                self.load_network(fn)

                for _ in range(n_random_pixels):                    
                    tile_ind = np.random.randint(0, len(self.dataset.tiles)-1)
                    tile_name = self.dataset.tiles[tile_ind]
                    nts = self.dataset.data[tile_ind][-1]   # type: ignore
                    pix_ind = np.random.randint(0, len(nts)-1)
                    
                    (y, x, timeless, timespans, prd_dates, y_dates, prd_dates, valid_values_x, valid_values_y) = self.dataset.get_one_pixel_timeseries(tile_ind, pix_ind)
                    nts = int(x.shape[0])
                    prd = self.model(torch.tensor(x), torch.tensor(timeless.squeeze()).expand((nts, -1)), torch.tensor(timespans)).detach()
                    prd = prd.squeeze().numpy()
                    
                    y_prd = prd[valid_values_x]
                    y_obs = y[12:]
                    
                    mae = (np.abs(y_obs - y_prd)).mean()
                    rmse = np.sqrt(np.mean((y_obs - y_prd)**2))
                    r2 = 1-np.var(y_obs - y_prd) / np.var(y_obs)

                    fig, ax = plt.subplots(figsize=(12,6))
                    ax.plot(y_dates, y, 'o', label='Observed', color='red', markersize=4, alpha=0.5)
                    ax.plot(prd_dates, prd, '-', label='Predicted', color='blue')
                    ax.set_title(f"Band {band_name}, Tile {tile_name}, Pixel {pix_ind}")
                    ax.set_xlabel("Date")
                    ax.set_ylabel("Reflectance")
                    ax.text(0.05, 0.95, f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}", transform=ax.transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                    ax.legend()

                    yield fig

    def draw_timeseries_2(self, tile_ind: int, pix_ind: int,
                        models:list[str|Path]| str|Path):
        # tile_ind=5; pix_ind=100
        # models = 'cfc_v6_b1_epoch-090.ckpt'
        if isinstance(models, (str, Path)):
            models = [Path(fld_checkpoints)/models]
        else:
            models = [Path(fld_checkpoints)/m for m in models]

        fns = pandas.DataFrame([dict(fn=fn, band=int(fn.stem.split('_')[2][1:])) for fn in models]) 
        for band in fns['band'].unique():
            # band=1
            band_name = bands_prefix[band].split('_')[0].upper()
            self.load_dataset(band, prepare_all_cases=True)            

            # pixel_indices = self.dataset.pixel_indices
            # ts_inds = np.where((pixel_indices[:,0]==tile_ind) & (pixel_indices[:,1]==pix_ind))[0]
            # if len(ts_inds)==0:
            #     raise ValueError(f"Tile {self.dataset.tiles[tile_ind]}, Pixel {pix_ind} not found in the dataset")

            # ts_inds = pixel_indices[ts_inds,2]
            # tile_data = self.dataset.all_tiles[tile_ind]
            # y = tile_data[0][ts_inds]
            # x = tile_data[1][ts_inds]
            # tl = tile_data[2][ts_inds]
            # ts = tile_data[3][ts_inds]

            fns_band = fns[fns['band']==band]['fn'].tolist()
            for fn in fns_band:
                # fn = fns_band[0]
                self.load_network(fn)

                (y, x, timeless, timespans, prd_dates, y_dates, prd_dates, valid_values_x, valid_values_y) = self.dataset.get_one_pixel_timeseries(tile_ind, pix_ind)
                nts = int(x.shape[0])
                prd = self.model(torch.tensor(x), torch.tensor(timeless.squeeze()).expand((nts, -1)), torch.tensor(timespans)).detach()
                prd = prd.squeeze().numpy()
                
                y_prd = prd[valid_values_x]
                y_obs = y[12:]
                
                mae = (np.abs(y_obs - y_prd)).mean()
                mse = np.mean((y_obs - y_prd)**2)
                r2 = 1-np.var(y_obs - y_prd) / np.var(y_obs)

                fig, ax = plt.subplots(figsize=(12,6))
                ax.plot(y_dates, y, 'o', label='Observed', color='red', markersize=4, alpha=0.5)
                ax.plot(prd_dates, prd, '-', label='Predicted', color='blue')
                ax.set_title(f"Band {band_name}, Tile {self.dataset.tiles[tile_ind]}, Pixel {pix_ind}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Reflectance")
                ax.text(0.05, 0.95, f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nR2: {r2:.4f}", transform=ax.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
                ax.legend()

                yield fig

    def make_predictions(self, models:list[str|Path]| str|Path, 
                         tiles: list[str]| str,                         
                         dates: list[YearMonth]| YearMonth,
                         out_folder: Path | str = fld_tiffs
                         ):
        # models = 'cfc_v6_b1_epoch-015.ckpt'
        # tiles = ['090W_49N', '055W_06S','015E_43N']
        # dates = [YearMonth(2023, 7), YearMonth(2023, 8)]
        # out_folder = fld_tiffs
        if isinstance(models, (str, Path)):
            fns_models:list[Path] = [Path(fld_checkpoints)/models]
        else:
            fns_models:list[Path] = [Path(fld_checkpoints)/m for m in models]
        if isinstance(tiles, str):
            tiles = [tiles]
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        #n_bands = 1 #landsat_data.shape[0] // (23 * len(years))
        #n_output_bands = 1
        n_timeless_features = 3 # gtemp_min, gtemp_max, gtemp at prediction date
        n_features = 3
        sequence_length = 12
        
        # This doesn't load any data. It just prepares some auxilary variables.
        ds = ArcoV2DatasetV6(
                        None, # not used
                        years, 
                        sequence_length,
                        limit=None, # limit is not used
                        band=0, # band is not used here
                        dtype=torch.float32, # not used
                        )

        time0 = time.time()
        for tile in tiles:
                gc.collect()
                # tile = '090W_49N'
                time1 = time.time()
                ttprint(f"Processing tile: {tile}")
                # load all data for the tile
                # TODO: load only needed bands (new class ?)
                (success, error, eta), meta, valid_data, data = cfc_sample.get_tile_data(tile)    
                (n_valid_pixels, inds_valid_pixels, valid_values_mask) = valid_data
                (landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy) = data

                profile=dict(
                        driver='GTiff',
                        count=1,
                        dtype='uint16',
                        width=utils.x_size,
                        height=utils.y_size,
                        crs=meta[0],
                        transform=meta[1],
                        nodata=65535,
                        blockxsize=1024, 
                        blockysize=1024,    
                        tiled=True,
                        compress='deflate',
                        predictor=2
                    )
                nodata = profile['nodata']
                
                ttprint(f'Tile {tile} loaded in {(time.time()-time1)/60:.2f} minutes')

                for fn_model in fns_models:
                    gc.collect()
                    # fn_model = fns_models[0]
                    ttprint(f"Processing model: {fn_model.name}")
                    # Check if the model file is in the checkpoints folder, otherwise assume it's a full path
                    if len(fn_model.parents) == 1:
                        fn_model = Path(fld_checkpoints)/fn_model
                    
                    band = int(fn_model.stem.split('_')[2][1:])

                    # Load model and compile
                    model = CfcLearnerV6.load_from_checkpoint(fn_model, map_location='cpu', strict=True) # input_size=input_size, sequence_length=12, output_size=output_size)
                    model = model.to(torch.bfloat16)
                    model.freeze()
                    model = torch.compile(model, backend='openvino')

                    time1 = time.time()
                    
                    for yearmonth in dates:
                        gc.collect()
                        # yearmonth = dates[0]
                        time2 = time.time()

                        year, month = yearmonth
                        ttprint(f"Preparing data for {year}-{month:02d}")

                        date = datetime.datetime(year, month, 15)
                        day_from_start = (date - datetime.datetime(years[0], 1, 1)).days - 1
                        n_pixels = landsat_data.shape[1]
                        n_dates = ds.days_from_start.shape[0]

                        timespans = np.empty((n_pixels, sequence_length), dtype=np.float32)
                        x = np.empty((n_pixels, sequence_length, n_features), dtype=np.float32)
                        timeless = np.empty((n_pixels, n_timeless_features), dtype=np.float32)
                        valid_pixels_ind = np.ones(n_pixels, dtype=bool)

                        # prepare all data for all pixels                        
                        @njit(parallel=True, fastmath=True, cache=True)
                        def _process_one_image(day_from_start, n_pixels, band, 
                                            x, timeless, timespans, valid_pixels_ind,
                                            days_from_start, 
                                            landsat_data, 
                                            modis_data,
                                            geom_temp_doy,
                                            valid_values_mask,
                                            sequence_length):
                            n_dates = days_from_start.shape[0]   
                            #geom_temp_min = geom_temp_doy.min(axis=0)
                            #geom_temp_max = geom_temp_doy.max(axis=0)                         
                            for pix in prange(n_pixels):
                                valid_values = valid_values_mask[:,pix]           
                                pix_dfs = days_from_start[valid_values]
                                ind = np.nonzero(pix_dfs < day_from_start)[0][-sequence_length:]
                                dfs = pix_dfs[ind]
                                if len(ind) < sequence_length:
                                    valid_pixels_ind[pix] = False
                                    continue
                                
                                x[pix, :, 0] = landsat_data[band*n_dates:(band+1)*n_dates, pix][valid_values][ind]
                                x[pix, :, 1] = modis_data[valid_values, pix][ind]
                                ind_doys = dfs % 23
                                x[pix, :, 2] = geom_temp_doy[ind_doys, pix]
                                timespans[pix, :-1] = dfs[1:] - dfs[:-1]
                                timespans[pix, -1] = day_from_start - dfs[-1]
                                
                                #timeless[pix, 0] = geom_temp_min[pix]   # gtemp_min
                                #timeless[pix, 1] = geom_temp_max[pix]   # gtemp_max

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

                        _process_one_image(day_from_start, n_pixels, band,
                            x, timeless, timespans, valid_pixels_ind,
                            ds.days_from_start.numpy(),
                            landsat_data, modis_data, geom_temp_doy,
                            valid_values_mask,
                            sequence_length)
                        
                        timeless[:,0] = geom_temp_doy.min(axis=0)   # gtemp_min
                        timeless[:,1] = geom_temp_doy.max(axis=0)   # gtemp_max

                        if not valid_pixels_ind.all():
                            x = x[valid_pixels_ind, :, :]
                            timeless = timeless[valid_pixels_ind, :]
                            timespans = timespans[valid_pixels_ind, :]
                        
                        timespans /= 366.0

                        x = torch.tensor(x, dtype=torch.bfloat16)
                        timeless = torch.tensor(timeless, dtype=torch.bfloat16)
                        timespans = torch.tensor(timespans, dtype=torch.bfloat16)

                        ttprint(f"Data for {year}-{month:02d} prepared in {(time.time()-time2)/60:.2f} minutes, running prediction for {x.shape[0]} pixels")

                        time3 = time.time()
                        with torch.no_grad():
                            prd = model(x, timeless, timespans)
                        prd = prd.to(torch.float16).numpy().squeeze()
                        
                        prd = (np.clip(prd, 0, 1) * 40000).astype(np.uint16)
                        if not valid_pixels_ind.all():
                            img = np.full(n_pixels, nodata, dtype=np.uint16)
                            img[valid_pixels_ind] = prd
                            img = img.reshape((utils.y_size, utils.x_size))
                        else:
                            img = prd.reshape((utils.y_size, utils.x_size))

                        ttprint(f"Prediction for {year}-{month:02d} done in {(time.time()-time3)/60:.2f} minutes, saving to {out_folder}")

                        # Saving prediction
                        time3 = time.time()                        
                        band_name = utils.bands_prefix_out[band]
                        fn = out_folder / f"{tile}_{year}{month:02d}_{band_name}.tif"

                        with rasterio.open(fn, 'w', **profile) as dst:
                            dst.write(img, 1)
                        ttprint(f"File {fn} saved in {(time.time()-time3)/60:.2f} minutes")
                        ttprint(f"Total time for {year}-{month:02d}: {(time.time()-time2)/60:.2f} minutes")
                    ttprint(f"Total time for tile {tile}: {(time.time()-time1)/60:.2f} minutes")
                ttprint(f"Total time elapsed: {(time.time()-time0)/60:.2f} minutes")

        
#%%
if __name__ == "__main__":
    '''
    RuntimeError: Too many open files. Communication with the workers is no longer possible. 
    Please increase the limit using `ulimit -n` in the shell or change the sharing strategy 
    by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code
    '''
    torch.multiprocessing.set_sharing_strategy('file_system')
    tester = CfcV6Test(fn_zarr=fn_zarr, 
                       fn_log="cfc_v6_test.log", 
                       device='cpu', 
                       dtype='float32',
                       limit=[120, 160],
                       percent_pixel=0.05,
                       ncases_validation=0.2,
                       #ncases=int(10e6)
                       )

    models = list(Path(fld_checkpoints).glob("cfc_v6_*.ckpt"))
    models = ['/mnt/nibble/gen_cog/arcov2/v6/checkpoints/best/cfc_v6_b1_epoch-038.ckpt',
              '/mnt/nibble/gen_cog/arcov2/v6/checkpoints/best/cfc_v6_b2_epoch-054.ckpt']
    fld_tiffs = Path("/mnt/nibble/gen_cog/arcov2/v6/predictions_finetuned")

    tiles = ['090W_49N', '055W_06S','015E_43N']
    dates = [YearMonth(year, month) for year in [2023] for month in range(1, 13)]

    tester.make_predictions(
        models=models,
        tiles=tiles,
        dates=dates,
        out_folder=fld_tiffs
    )

    #tester.run_all(fld_checkpoints=fld_checkpoints)
    # tester.test_one_model(fn_checkpoint=Path(fld_checkpoints)/"cfc_v6_b1_epoch-094.ckpt", band=1)

    # for fig in tester.draw_timeseries(n_random_pixels=5, models=['cfc_v6_b0_epoch-026.ckpt']):
    #     fig.show()
    #     #fig.savefig(f"test_{time.time()}.png", dpi=150)
    #     plt.pause(0.1)
    #     plt.close(fig)