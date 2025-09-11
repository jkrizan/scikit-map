#%%
#%%
from settings import bands_prefix
from pathlib import Path
import pandas
import torcheval.metrics
from cfc_v6 import ArcoV2DatasetV6, CfcLearnerV6
import numpy as np  
import torch, torcheval
from torch.utils.data import DataLoader
from utils import ttprint
import time
import tqdm
import pandas
import matplotlib.pyplot as plt

fn_zarr = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
years = np.arange(2000, 2024)
sequence_length = 12

fld_checkpoints = "/mnt/nibble/gen_cog/arcov2/v6/checkpoints"
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
            self.fn_log.write_text("timestamp\tfn_checkpoint\tband\tmae\tmse\r2\n")
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
        r2 = torcheval.metrics.functional.r2_score(y, prdy).item()
        ttprint(f"MAE: {mae}, MSE: {mse}, R2: {r2}")
        with self.fn_log.open("a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{fn_checkpoint.name}\t{band}\t{mae}\t{mse}\t{r2}\n")

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
        fns_checkpoints = sorted(fld_checkpoints.glob("cfc_v6_b*_epoch-*.ckpt"))
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
                       percent_pixel=0.1,
                       ncases_validation=0.2,
                       ncases=int(10e6))
    
    # tester.run_all(fld_checkpoints=fld_checkpoints)
    for fig in tester.draw_timeseries(n_random_pixels=5, models=['cfc_v6_b1_epoch-081.ckpt']):
        fig.show()
        #fig.savefig(f"test_{time.time()}.png", dpi=150)
        plt.pause(0.1)
        plt.close(fig)