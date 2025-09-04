#%%
import cfc_v6
import numpy as np  
import torch

fn_zarr = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
years = np.arange(2000,2024)
sequence_length = 12
limit=None
band=0
dataset = cfc_v6.ArcoV2DatasetV6(fn_zarr, years, sequence_length, band, limit=limit, 
                                 percent_pixels=1, device='cpu', dtype=torch.float16)
print(f"Dataset length: {len(dataset)}")

dataset.prepare_all_cases()
#%%
# y, x, gtemp, timespans = dataset[0]