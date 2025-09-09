#%%
#%%
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

    def load_dataset(self, band: int):
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
                       limit=None,
                       percent_pixel=0.1,
                       ncases_validation=0.2,
                       ncases=int(10e6)
    )
    tester.run_all(fld_checkpoints=fld_checkpoints)