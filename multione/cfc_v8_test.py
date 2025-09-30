#%%
#%%
from collections import namedtuple

from settings import bands_prefix
import utils, cfc_sample
from utils import ttprint
from pathlib import Path
import pandas
import torcheval.metrics
from cfc_v8 import ArcoV2DatasetV8, CfcLearnerV8, MemoryDataLoader
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
import yaml

fn_zarr = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
years = np.arange(2000, 2024)
sequence_length = 12

fld_checkpoints = "/mnt/nibble/gen_cog/arcov2/v8/checkpoints"
fld_tiffs = "/mnt/nibble/gen_cog/arcov2/v8/predictions"

YearMonth = namedtuple('YearMonth', ['year', 'month'])
#%%
class CfcV8Test:

    def __init__(self, 
                 fn_zarr: Path | str,   
                 fn_log: Path | str | None,              
                 device: str = 'cpu',
                 dtype: str = 'float32',
                 limit: int| None = None,
                 percent_pixel: float = 0.1,
                 ncases_validation: float = 0.1, 
                 ncases: int = int(10e6)
                ) -> None:
        
        self.fn_zarr = fn_zarr
        self.fn_log = Path(fn_log) if fn_log is not None else None
        self.limit = limit
        self.percent_pixels = percent_pixel
        self.ncases_validation = ncases_validation
        self.ncases = ncases

        bands_names = [b.split('_')[0] for b in bands_prefix[:6]]
        if fn_log is not None and not self.fn_log.exists():
            self.fn_log.parent.mkdir(parents=True, exist_ok=True)
            header = "timestamp\tfn_checkpoint\tversion\tdirection\tmae\trmse\tr2" + \
                "\tactivation\tbackbone_layers\thidden_size\tlr\t" + \
                "\t".join([f"mae_{b}" for b in bands_names]) + \
                "\t" + "\t".join([f"rmse_{b}" for b in bands_names]) + \
                "\t" + "\t".join([f"r2_{b}" for b in bands_names]) + "\n"
            self.fn_log.write_text(header)
        self.device = device
        if dtype=='float32':
            self.dtype = torch.float32
        elif dtype=='float16':
            self.dtype = torch.float16
        elif dtype=='bfloat16':
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

    def load_network(self, fn_checkpoint: Path | str):
        learner = CfcLearnerV8.load_from_checkpoint(fn_checkpoint)
        model = learner.model.to(self.device).to(self.dtype)
        #model.freeze()
        model.eval().compile(fullgraph=True)
        self.model = model

    def test_one_model(self, fn_checkpoint: Path, version: int):
        # cfc_v8_b1_epoch-087.ckpt

        self.load_network(fn_checkpoint)
        hparams = yaml.load((fn_checkpoint.parent.parent / "hparams.yaml").read_text(), Loader=yaml.Loader)
        ttprint(f"Model loaded: {fn_checkpoint}, {hparams}")
        

        y=[]; prdy=[]
        time0 = time.time()
        for i, (yb, xb, tlb, tsb) in tqdm.tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            # (yb, xb, tlb, tsb) = next(iter(self.dataloader))
            y.append(yb)
            prd = self.model(xb, tlb, tsb).detach()            
            prdy.append(prd)        
        y = torch.cat(y, dim=0)
        prdy = torch.cat(prdy, dim=0)
        print(f"Evaluation done in {time.time()-time0:.1f} seconds")

        residuals = y - prdy
        for direction in ('fwd', 'bwd'):
            if direction == 'fwd':
                dim2 = 0
            else:
                dim2 = 1
            res = residuals[:,:,dim2].squeeze()
            mae = torch.abs(res).mean(dim=0)
            mse = (res ** 2).mean(dim=0)
            rmse = torch.sqrt(mse)
            r2 = torcheval.metrics.functional.r2_score(y[:,:,dim2], prdy[:,:,dim2], multioutput='raw_values')
            ttprint(f"MAE: {mae.mean()}, MSE: {mse.mean()}, RMSE: {rmse.mean()}, R2: {r2.mean()}")
            with self.fn_log.open("a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{fn_checkpoint.name}\t{version}\t{direction}\t{mae.mean()}\t{rmse.mean()}\t{r2.mean()}")
                f.write(f"\t{hparams['activation']}\t{hparams['backbone_layers']}\t{hparams['hidden_size']}\t{hparams['lr']}")
                for b in range(mae.shape[0]):
                    f.write(f"\t{mae[b].item()}")
                for b in range(rmse.shape[0]):
                    f.write(f"\t{rmse[b].item()}")
                for b in range(r2.shape[0]):
                    f.write(f"\t{r2[b].item()}")
                f.write("\n")
        return
    
    def load_dataset(self, prepare_all_cases: bool = True):
        self.dataset = ArcoV2DatasetV8(self.fn_zarr, 
                                       years, 
                                       sequence_length, 
                                       bands=[0,1,2,3,4,5],
                                       limit=self.limit,
                                       percent_pixels=self.percent_pixels, 
                                       device='cpu', 
                                       dtype=self.dtype)
        ttprint(f"Dataset length: {len(self.dataset)}")

        if prepare_all_cases:
            self.dataset.prepare_all_cases()
        #self.dataloader = DataLoader(self.dataset, batch_size=4096, shuffle=False, num_workers=8)
        self.dataloader = MemoryDataLoader(self.dataset, batch_size=4096, indexes=None)

    def run_all(self, fld_checkpoints: Path | str):
        fld_checkpoints = Path(fld_checkpoints)
        fns_checkpoints = sorted(fld_checkpoints.glob("**/v8_*.ckpt"))
        fns = pandas.DataFrame([dict(fn=fn, version=int(fn.parent.parent.stem.split('_')[1])) for fn in fns_checkpoints])
        #bands = fns['band'].unique()
        ttprint(f"Found {len(fns_checkpoints)} checkpoints.") # " for bands: {bands}")

        df = pandas.read_csv(self.fn_log, sep="\t")
        fns_done = df['fn_checkpoint'].unique().tolist()
        fns = fns[~fns['fn'].apply(lambda x: x.name in fns_done)]
        ttprint(f"{len(fns_done)} checkpoints have been tested, {len(fns)} remaining")

        self.load_dataset()

        for (fn_checkpoint, version) in tqdm.tqdm(fns[['fn','version']].itertuples(index=False), total=len(fns)):
            # (fn_checkpoint, version) = next(fns[['fn','version']].itertuples(index=False))
            ttprint(f"Testing checkpoint: {version=}, {fn_checkpoint.name=}")
            self.test_one_model(fn_checkpoint, version)

    
    def draw_timeseries(self, tile_ind: int, pix_ind: int,
                        models:list[str|Path]| str|Path):
        # tile_ind=3; pix_ind=100
        # models = models=['v8_21_epoch=022.ckpt']
        if isinstance(models, (str, Path)):
            models = [Path(fld_checkpoints)/models]
        else:
            models = [Path(fld_checkpoints)/m for m in models]


        for fn in models:
            # fn = models[0]
            self.load_network(fn)

            (directions, y, x, timeless, timespans, dates, valid_values) = self.dataset.get_one_pixel_timeseries(tile_ind, pix_ind)
            #inds = self.dataset.get_one_pixel_indices(tile_ind, pix_ind)
            # inds = [1202496, 1202497, 1202498, 1202499, 1202500, 1202501, 1202502]
            #(y, x, timeless, timespans) = self.dataset.get_cases(inds)
            # x = x.permute(0,2,1)  # (n_dates, seq_len, n_features)
            y_prd = self.model.forward(x, timeless, timespans).detach().numpy()
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

    def log(self, message: str):
        with self.fn_log.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{message}\n")

    def make_predictions(self, model_name: str, 
                         tiles: list[str]| str,                         
                         dates: list[YearMonth]| YearMonth,
                         out_folder: Path | str = fld_tiffs,
                         fn_log: Path | str = "cfc_v8_predictions.log"
                         ):
        # models = 'cfc_v6_b1_epoch-015.ckpt'
        # tiles = ['090W_49N', '055W_06S','015E_43N']
        # dates = [YearMonth(2023, 7), YearMonth(2023, 8)]
        # out_folder = fld_tiffs
        self.fn_log = Path(fn_log)
        if not self.fn_log.exists():
            self.fn_log.parent.mkdir(parents=True, exist_ok=True)
            header = "timestamp\taction\ttile\tdate\tn_pixels\tduration\n"
            self.fn_log.write_text(header)

        # if isinstance(models, (str, Path)):
        #     fns_models:list[Path] = [Path(fld_checkpoints)/models]
        # else:
        #     fns_models:list[Path] = [Path(fld_checkpoints)/m for m in models]
        fn_model = Path(fld_checkpoints)/model_name

        if isinstance(tiles, str):
            tiles = [tiles]
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        #n_bands = 1 #landsat_data.shape[0] // (23 * len(years))
        #n_output_bands = 1
        n_timeless_features = 3 # gtemp_min, gtemp_max, gtemp at prediction date
        n_features = 8 # landsat, modis, geom_temp_doy
        sequence_length = 12
        bands = np.arange(6)  # bands used for prediction
        
        # This doesn't load any data. It just prepares some auxilary variables.
        ds = ArcoV2DatasetV8(
                        None, # not used
                        years, 
                        sequence_length,
                        bands = [0,1,2,3,4,5], # not used
                        limit=None, # limit is not used                        
                        dtype=torch.float32, # not used
                        )

        # Load model and compile
        ttprint(f"Processing model: {fn_model.name}")
        model = CfcLearnerV8.load_from_checkpoint(fn_model, map_location='cpu', strict=True) # input_size=input_size, sequence_length=12, output_size=output_size)
        model = model.to(torch.bfloat16)
        model.freeze()
        model = torch.compile(model, backend='openvino')

        time0 = time.time()
        for tile in tiles:
                gc.collect()
                # tile = '090W_49N'
                time1 = time.time()
                ttprint(f"Processing tile: {tile}")
                # load all data for the tile
                                
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

                duration = (time.time()-time1)
                ttprint(f'Tile {tile} loaded in {duration/60:.2f} minutes')
                self.log(f'load\t{tile}\t{n_valid_pixels}\t{duration:.2f}')
                                    
                # Check if the model file is in the checkpoints folder, otherwise assume it's a full path
                # if len(fn_model.parents) == 1:
                #     fn_model = Path(fld_checkpoints)/fn_model                                                   

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
                    # n_dates = ds.days_from_start.shape[0]

                    fns = list(out_folder.glob(f"{tile}_{year}{month:02d}_*.tif"))
                    if len(fns) == len(bands):
                        ttprint(f"Predictions for {tile} {year}-{month:02d} already exist, skipping")
                        self.log(f'skip\t{tile}\t{year}-{month:02d}\t{len(bands)}\t0.00')
                        continue
                        
                    timespans = np.empty((n_pixels, sequence_length + 1), dtype=np.float32)
                    x = np.empty((n_pixels, sequence_length, n_features), dtype=np.float32)
                    timeless = np.empty((n_pixels, n_timeless_features), dtype=np.float32)
                    valid_pixels_ind = np.ones(n_pixels, dtype=bool)

                    # prepare all data for all pixels                        
                    @njit(parallel=True, fastmath=True, cache=True)
                    def _process_one_image(day_from_start, n_pixels, bands, 
                                        x, timeless, timespans, valid_pixels_ind,
                                        days_from_start, 
                                        landsat_data, 
                                        modis_data,
                                        geom_temp_doy,
                                        valid_values_mask,
                                        sequence_length):
                        n_dates = days_from_start.shape[0]   
                        n_bands = len(bands)
                                                    
                        for pix in prange(n_pixels):
                            valid_values = valid_values_mask[:,pix]           
                            pix_dfs = days_from_start[valid_values]
                            ind = np.nonzero(pix_dfs < day_from_start)[0][-sequence_length:]
                            dfs = pix_dfs[ind]
                            if len(ind) < sequence_length:
                                valid_pixels_ind[pix] = False
                                continue
                            for b in bands:
                                x[pix, :, b] = landsat_data[b*n_dates:(b+1)*n_dates, pix][valid_values][ind]
                            x[pix, :, n_bands] = modis_data[valid_values, pix][ind]
                            ind_doys = dfs % 23
                            x[pix, :, n_bands+1] = geom_temp_doy[ind_doys, pix]
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

                    _process_one_image(day_from_start, n_pixels, bands,
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

                    duration = (time.time()-time2)
                    ttprint(f"Data for {year}-{month:02d} prepared in {duration/60:.2f} minutes, running prediction for {x.shape[0]} pixels")
                    self.log(f'prepare\t{tile}\t{year}-{month:02d}\t{x.shape[0]}\t{duration:.2f}')

                    time3 = time.time()
                    model.eval()
                    with torch.no_grad():
                        prd = model(x, timeless, timespans)
                    prd = prd.to(torch.float16).numpy().squeeze()
                    # TODO: need to select direction based on data availability
                    prd = prd[:,:,0]  # take forward direction 
                    prd = (np.clip(prd, 0, 1) * 40000).astype(np.uint16)

                    duration = (time.time()-time3)
                    ttprint(f"Prediction for {year}-{month:02d} done in {duration/60:.2f} minutes, saving to disk")
                    self.log(f'predict\t{tile}\t{year}-{month:02d}\t{x.shape[0]}\t{duration:.2f}')

                    time3 = time.time()
                    for bi,b in enumerate(bands):
                        # bi=0; b=bands[bi]
                        if not valid_pixels_ind.all():
                            img = np.full(n_pixels, nodata, dtype=np.uint16)
                            img[valid_pixels_ind] = prd[:,bi]
                            img = img.reshape((utils.y_size, utils.x_size))
                        else:
                            img = prd[:, bi].reshape((utils.y_size, utils.x_size))

                        # Saving prediction
                        
                        band_name = utils.bands_prefix_out[b].split('_')[0]
                        fn = out_folder / f"{tile}_{year}{month:02d}_{band_name}.tif"

                        with rasterio.open(fn, 'w', **profile) as dst:
                            dst._set_all_scales([1./40000.])
                            dst._set_all_offsets([0.0])
                            dst.write(img, 1)
                        # ttprint(f"File {fn} saved in {(time.time()-time3)/60:.2f} minutes")

                    duration = (time.time()-time3)
                    ttprint(f"Prediction for {year}-{month:02d} done in {duration/60:.2f} minutes, saving to {out_folder}")
                    self.log(f'save\t{tile}\t{year}-{month:02d}\t{len(bands)}\t{duration:.2f}')

                duration = (time.time()-time1)
                ttprint(f"Total time for tile {tile}: {duration/60:.2f} minutes")
                self.log(f'total\t{tile}\t{n_valid_pixels}\t{duration:.2f}')

                ttprint(f"Total time elapsed: {(time.time()-time0)/60:.2f} minutes")

#%%
def draw_timeseries_example():
    tester = CfcV8Test(fn_zarr=fn_zarr, 
                       fn_log=None, 
                       device='cpu', 
                       dtype='float32',
                       limit=[120, 160],
                       percent_pixel=0.1,
                       ncases_validation=0.2,                       
                       )
     
    tester.load_dataset(prepare_all_cases=False)
    ntiles = len(tester.dataset.tiles)

    for i in range(3):
        t = np.random.randint(ntiles)
        npix = len(tester.dataset.data[t][-1])
        p = np.random.randint(npix)

        fig = tester.draw_timeseries(t,p,'v8_21_epoch=022.ckpt')
        plt.show()



def make_predictions_3_tiles():
    tester = CfcV8Test(fn_zarr=fn_zarr, 
                       fn_log=None, 
                       device='cpu', 
                       dtype='float32',
                       limit=[120, 160],
                       percent_pixel=0.1,
                       ncases_validation=0.2,
                       #ncases=int(10e6)
                       )

    #models = list(Path(fld_checkpoints).glob("cfc_v8_*.ckpt"))
    # models = ['/mnt/nibble/gen_cog/arcov2/v8/checkpoints/best/cfc_v8_b1_epoch-038.ckpt',
    #           '/mnt/nibble/gen_cog/arcov2/v8/checkpoints/best/cfc_v8_b2_epoch-054.ckpt']
    #fld_tiffs = Path("/mnt/nibble/gen_cog/arcov2/v8/predictions_finetuned")

    model = "v1_e022"
    fld_out = Path("/mnt/nibble/gen_cog/arcov2/v8")
    fld_tiffs = Path(f"/mnt/nibble/gen_cog/arcov2/v8/predictions_{model}")

    tiles = ['055W_06S','015E_43N','090W_49N']
    #tiles = ['090W_49N']
    #dates = [YearMonth(year, month) for year in [2023] for month in range(1, 13)]
    dates = [YearMonth(year, month) for year in range(2023,2019,-1) for month in range(1, 13)]
    #dates = [YearMonth(year, month) for year in range(2023,2024) for month in range(1, 13)]

    tester.make_predictions(
        model_name=model + ".ckpt",
        tiles=tiles,
        dates=dates,
        out_folder=fld_tiffs,
        fn_log=fld_out/f"{model}_predictions.log"
    )


def statistics_all_models():    
    '''
    RuntimeError: Too many open files. Communication with the workers is no longer possible. 
    Please increase the limit using `ulimit -n` in the shell or change the sharing strategy 
    by calling `torch.multiprocessing.set_sharing_strategy('file_system')` at the beginning of your code
    '''
    torch.multiprocessing.set_sharing_strategy('file_system')
    tester = CfcV8Test(fn_zarr=fn_zarr, 
                       fn_log="cfc_v8_test.log", 
                       device='cpu', 
                       dtype='float32',
                       limit=[120, 160],
                       percent_pixel=0.1,
                       ncases_validation=0.2,
                       #ncases=int(10e6)
                       ) 

    tester.run_all(fld_checkpoints=fld_checkpoints)

def analyse_logs():
    df = pandas.read_csv("cfc_v8_test.log", sep="\t")
    #print(df.head())
    #print(df['fn_checkpoint'].value_counts())
    #print(df.groupby(['fn_checkpoint','direction'])['mae'].mean().unstack())
    #print(df.groupby(['fn_checkpoint','direction'])['rmse'].mean().unstack())
    #print(df.groupby(['fn_checkpoint','direction'])['r2'].mean().unstack())
    best_mae = df.groupby(['fn_checkpoint','direction'])['mae'].mean().unstack().min().min()
    best_rmse = df.groupby(['fn_checkpoint','direction'])['rmse'].mean().unstack().min().min()
    best_r2 = df.groupby(['fn_checkpoint','direction'])['r2'].mean().unstack().max().max()
    print(f"Best MAE: {best_mae}, Best RMSE: {best_rmse}, Best R2: {best_r2}")

    dff = df.groupby(['fn_checkpoint','direction'])['rmse'].mean().unstack()
    dff['mean_rmse'] = dff.mean(axis=1)
    dff = dff.sort_values('mean_rmse')
    print(dff.head(5))

    res = []
    for fn in dff.head(5).index:
        #print(df[df['fn_checkpoint']==fn].sort_values('direction'))
        mean_rmse = dff.loc[fn]['mean_rmse']        
        dfs = df[df['fn_checkpoint']==fn].iloc[0]
        res.append((fn, dfs['backbone_layers'], dfs['hidden_size'], dfs['lr'], mean_rmse, dfs['version']))

    dfr = pandas.DataFrame(res, columns=['fn_checkpoint','backbone_layers','hidden_size','lr', 'mean_rmse', 'version'])
    print(dfr)
#%%
if __name__ == "__main__":
    import sys
    arg = sys.argv[1]
    if arg == 'stats':
        statistics_all_models()
    elif arg == 'predict':
        make_predictions_3_tiles()
    
    
    
    #statistics_all_models()
    #make_predictions_3_tiles()

    # tester.test_one_model(fn_checkpoint=Path(fld_checkpoints)/"cfc_v6_b1_epoch-094.ckpt", band=1)

    # for fig in tester.draw_timeseries(n_random_pixels=5, models=['cfc_v6_b0_epoch-026.ckpt']):
    #     fig.show()
    #     #fig.savefig(f"test_{time.time()}.png", dpi=150)
    #     plt.pause(0.1)
    #     plt.close(fig)