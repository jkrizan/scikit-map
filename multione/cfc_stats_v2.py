#%%
import datetime
import numpy as np
import torch
from pathlib import Path
import datetime

from cfc_v4 import CfcLearner_v4, ArcoV2DatasetV2
import cfc_sample
from cfc_dataset import _process_one_pixel

#%%
#import importlib
#cfc_train = importlib.reload(cfc_train)
#%%
def test_timeseries():
#%%
    tile = '055W_06S'
    years = np.arange(2000, 2024)
    sequence_length = 12
    
    (success, error, eta), meta, valid_data, data = cfc_sample.get_tile_data(tile)    
    (n_valid_pixels, inds_valid_pixels, valid_values_mask) = valid_data
    (landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy) = data

    n_bands = landsat_data.shape[0] // (23 * len(years))
    n_output_bands = 7
    n_features = 9

    ds = ArcoV2DatasetV2(None,  years, sequence_length)

    pixel_ind = 55778
    valid_values = valid_values_mask[:, pixel_ind]
    nvv = valid_values.sum()
    nts = nvv - sequence_length
    n_dates = ds.days_from_start.shape[0]
    j_dates = ds.days_from_start[valid_values]
    j_lsdata = np.empty((n_bands, nvv), dtype=np.float32)
    for b in range(n_bands):
        j_lsdata[b,:] = landsat_data[b*n_dates:(b+1)*n_dates, pixel_ind][valid_values]
    j_msdata = modis_data[valid_values, pixel_ind]    
    j_gtemp = geom_temp_doy[ds.ind_doys[valid_values], pixel_ind]
    timeless = covariate_data[:, pixel_ind]

    y = np.empty((nts, n_output_bands), dtype=np.float32)
    timespans = np.empty((nts, sequence_length), dtype=np.float32)
    x = np.empty((nts, sequence_length, n_features), dtype=np.float32)  
    _process_one_pixel(y, x, timespans, j_dates, j_lsdata, j_msdata, j_gtemp, ds.sequence_length)

    x[np.isnan(x)] = 0
    x[:,:,7]  = x[:,:,7]/10000
    x[:,:,8]  = x[:,:,8]/100
                                           
#%%
    fn_ckpt = Path('/mnt/nibble/gen_cog/arcov2/cfc-v4_e-41.ckpt')
    input_size = 9 #dataset.n_features
    output_size = 7 #dataset.n_output_bands
    sequence_length = 12
    #n_timeless_features = 17 #dataset.n_timeless_features
    model = CfcLearner_v4.load_from_checkpoint(fn_ckpt, map_location='cpu', strict=True) # input_size=input_size, sequence_length=12, output_size=output_size)
    model.freeze()
#%%
    


#%%
    # tile_data = dataset[1][0]
    # y, x, _, timespans = tile_data
    # y, x, timespans = y.detach().clone(), x.detach().clone(), timespans.detach().clone()
    # x[:,:,7]  = x[:,:,7]/10000
    # x[:,:,8]  = x[:,:,8]/100

    xx = torch.tensor(x).unsqueeze(1)
    tt = torch.tensor(timespans).unsqueeze(1)
    y_hat = model(xx, tt )

    print(torch.nn.MSELoss()(y_hat, torch.tensor(y)).item())
    y_hat = y_hat.detach().numpy()  

#%%
    dates = ds.dates[valid_values]
    b=0
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(dates[sequence_length:], y_hat[:,b], '*', label='y_hat')
    ax.plot(dates[sequence_length:], y[:,b],'.', label='y')
    ax.legend()
    plt.show()
#%%

    



#%%
    