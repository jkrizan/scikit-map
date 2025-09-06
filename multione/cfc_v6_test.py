#%%
import cfc_v6
import numpy as np  
import torch

fn_zarr = "/mnt/nibble/gen_cog/arcov2/sample_v6.zarr"
years = np.arange(2000,2024)
sequence_length = 12
limit=None
band=1
dataset = cfc_v6.ArcoV2DatasetV6(fn_zarr, years, sequence_length, band, limit=limit, 
                                 percent_pixels=1, device='cpu', dtype=torch.float16)
print(f"Dataset length: {len(dataset)}")

dataset.prepare_all_cases()
dl = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=4)

#%%
# y, x, gtemp, timespans = dataset[0]
for i, (y, x, gtemp, timespans) in enumerate(dl):
    print(f"Batch {i:05}: y: {torch.isnan(y).sum()}, x: {torch.isnan(x).sum()}, gtemp: {torch.isnan(gtemp).sum()}, timespans: {torch.isnan(timespans).sum()}")

# %%
