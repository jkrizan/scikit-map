#%%
import datetime
import numpy as np
import torch
from pathlib import Path
import datetime

import cfc_dataset, cfc_sample, cfc_train, cfc


#%%
def test_timeseries():
#%%
    tile = '055W_06S'
    years = np.arange(2000, 2024)
    
    (success, error, eta), meta, valid_data, data = cfc_sample.get_tile_data(tile)    
    (n_valid_pixels, inds_valid_pixels, valid_values_mask) = valid_data
    (landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy) = data
    ds = cfc_dataset.ArcoV2Dataset(None, years, sequence_length)
    covariate_data[np.isnan(covariate_data)] = 0
    geom_temp_doy[np.isnan(geom_temp_doy)] = 0    
    npixels = landsat_data.shape[1]
    n_bands = landsat_data.shape[0] // (len(years)*23)  # number of bands per year
    n_features = n_bands + 2
    n_output_bands = 7
#%%
#%%
    fn_ckpt = Path('/mnt/nibble/gen_cog/arcov2/cfcv1_epoch10.ckpt')
    input_size = 9 #dataset.n_features
    output_size = 7 #dataset.n_output_bands
    n_timeless_features = 17 #dataset.n_timeless_features
    model = cfc.CfC(input_size,
                num_hidden_units=64,
                output_size=output_size,
                timeless_layers=[n_timeless_features, n_timeless_features//2, n_timeless_features//3],
                backbone_layers=[128, 64, 32],
                backbone_dropout=0.1
                )                
    hparams = {
            'lr': 0.01
            }
    model = cfc_train.ArcoV2Learner.load_from_checkpoint(fn_ckpt,map_location='cpu',strict=False, model=model, hparams=hparams)
    model = model.eval()
    model.freeze()
#%%
    sequence_length = 12
    pixel_ind = 55779
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

    y = np.empty((nts, n_output_bands), dtype=np.float32)
    timespans = np.empty((nts, sequence_length), dtype=np.float32)
    x = np.empty((nts, sequence_length, n_features), dtype=np.float32)
    x_timeless = covariate_data[:,pixel_ind]

    cfc_dataset._process_one_pixel(y, x, timespans, j_dates, j_lsdata, j_msdata, j_gtemp, ds.sequence_length)
    x[np.isnan(x)] = 0


#%%
    y_hat,_ = model.forward(torch.tensor(x),
                          torch.tensor(x_timeless).unsqueeze(0),
                         torch.tensor(timespans)
                         )
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
    year = 2010
    doy = 15
    date = (datetime.datetime(year, 1, 1) + datetime.timedelta(doy - 1)).date()

    ind_before = np.where(ds.dates < date)[0][-sequence_length:]
    
    xx_timeless = covariate_data.T 
    ttimespans = np.empty((npixels, sequence_length), dtype=np.float32)
    for i in range(npixels):
        ttimespans[i,:] = 
    xx = np.empty((npixels, sequence_length, n_features), dtype=np.float32)
    #np.empty((npixels, n_timeless_features), dtype=np.float32)
    



#%%
    