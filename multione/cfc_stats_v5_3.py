#%%
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import datetime
import numpy as np
from numpy.ma import true_divide
from pystac import cache
from sklearn.metrics import dcg_score
import torch
from torchmetrics.regression import R2Score
from pathlib import Path
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from cfc_v5_3 import CfcLearner_v5, ArcoV2DatasetV3
import cfc_sample
from cfc_dataset import _process_one_pixel
from settings import bands_prefix_out
import utils
from utils import get_temperature_for_doy, get_temperature_for_year
from numba import njit, prange
import matplotlib.pyplot as plt
import rasterio
import tqdm
import seaborn as sns
import pandas
from PIL import Image, ImageFont
import io
import nncf
import gc

#import torch; import intel_extension_for_pytorch as ipex
import openvino as ov

fld_out = Path('/mnt/nibble/gen_cog/arcov2/prd_cfcv5.3')
fld_out.mkdir(exist_ok=True, parents=True)
fn_ckpt = Path('/mnt/nibble/gen_cog/arcov2/cfc_v5.3-e_64.ckpt')
fld_gifs = Path('/mnt/nibble/gen_cog/arcov2/gifs_v5.3')
fld_gifs.mkdir(exist_ok=True, parents=True)
#%%
def statistics():
#%%
    ds = ArcoV2DatasetV3(Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.zarr"),
                         np.arange(2000, 2024), 12, limit=10, read_timeless=False, dtype=torch.bfloat16)
    #_, vds = ds.get_train_validation_subset(0.2)
    # rndgen=np.random.default_rng(43)
    # inds = np.arange(len(ds)); rndgen.shuffle(inds)
    # valprc = 0.2; 
    #vds = Subset(ds, inds[:int(len(ds)*valprc)])
    dl = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=8, prefetch_factor=3)

    ## openvino ########################
    #model = CfcLearner_v5.load_from_checkpoint(fn_ckpt, precision='16-mixed') # input_size=input_size, sequence_length=12, output_size=output_size)
    #model = model.to(torch.bfloat16)
    #model = torch.compile(model, backend='openvino')
    
    ## openvino quantization #########################
    nncf_config_dict = {
        # "model": "alexnet",
        "pretrained": 1,
        "input_info": [ {"sample_size": [8192, 1, 12, 8] },
                        {"sample_size": [8192, 1, 12] },                        
        ],
    }
    model = CfcLearner_v5.load_from_checkpoint(fn_ckpt) # input_size=input_size, sequence_length=12, output_size=output_size)
    example_input = next(iter(dl)); y,x,timespans = example_input
    #fx_model = torch.export.export_for_training(model.model.eval(), args=(x.to(torch.float32),timespans.to(torch.float32))).module()
                                                #input_info=nncf_config_dict['input_info']).module()
    #quantization_dataset = nncf.Dataset(dl, lambda batch: dict(x=batch[1].to(torch.float32), timespans=batch[2].to(torch.float32)))
    def trans_func(batch):
        return (batch[1].to(torch.float32), batch[2].to(torch.float32))
    
    quantization_dataset = nncf.Dataset(dl, trans_func)
    quantized_model = nncf.quantize(model.eval(), quantization_dataset, 
                                       preset=nncf.QuantizationPreset.PERFORMANCE, # "mixed"  #"performance"                                       
                                       target_device=nncf.TargetDevice.CPU) 
    #model_export = torch.export.export_for_inference(quantized_model.model, args=(x.to(torch.float32),timespans.to(torch.float32))) #.module()
    ov_model = ov.convert_model(quantized_model.model,
                                input=dict(x=[-1,1,12,8],timespans=[-1,1,12]), 
                                example_input=dict(x=x.to(torch.float32), timespans=timespans.to(torch.float32)))
    ov.save_model(ov_model, fn_ckpt.parent/f'{fn_ckpt.stem}_quant.xml')
    #quantized_model_compiled = torch.compile(quantized_model)
    

    ## ipex #################
    #submodel = model.model
    #submodel = submodel.eval()
    # model.freeze()
    # m = ipex.optimize(submodel, 
    #                       dtype=torch.float32, 
    #                       replace_dropout_with_identity=True,
    #                       #election = True
    #                       )
    # m.compile()
    # this model (m) crash the kernel when evaluated

#%%
    ds = ArcoV2DatasetV3(Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.zarr"),
                         np.arange(2000, 2024), 12, limit=10, read_timeless=False, dtype=torch.bfloat16)
    dl = DataLoader(ds, batch_size=8192, shuffle=False, num_workers=8, prefetch_factor=3)
    #model = quantized_model_compiled
    model = ov.compile_model(fn_ckpt.parent/f'{fn_ckpt.stem}_quant.xml')

    y=[]; prdy=[]
    for i, (yb, xb, tsb) in tqdm.tqdm(enumerate(dl), total=len(dl)): #tqdm.tqdm(dl): #
        # (yb, xb, tsb) = next(iter(dl))        
        y.append(yb)
        prd = model((xb.to(torch.float32).unsqueeze(1), tsb.to(torch.float32).unsqueeze(1)))
        prdy.append(prd[0])

    y = torch.cat(y, dim=0)
    prdy = torch.tensor(np.concatenate(prdy, axis=0))

    print("Statistics:")
    print(f"  - MAE: {(torch.abs(y - prdy)).mean(dim=0)}")
    print(f"  - MSE: {(torch.mean((y - prdy) ** 2, dim=0))}")
    print(f"  - R2: {(1 - torch.var(y - prdy, dim=0) / torch.var(y, dim=0))}")

    df = pandas.DataFrame()
    for b in range(y.size(1)):
        df[f"band_{b}_observed"] = y[:,b].detach().to(torch.float32).numpy()
        df[f"band_{b}_predicted"] = prdy[:,b].detach().to(torch.float32).numpy()

    df.to_pickle(fn_ckpt.parent/f'{fn_ckpt.stem}_observed_vs_predicted.pickle')
    # b=0
    # sns.regplot(x=y[:,b].detach().numpy(), 
    #             y=prdy[:,b].detach().numpy(), 
    #             line_kws={"color": "red"})

#%%
def test_timeseries():
#%%

    tile = '055W_06S'
    years = np.arange(2000, 2024)
    sequence_length = 12
    
    (success, error, eta), meta, valid_data, data = cfc_sample.get_tile_data(tile)    #TODO: get_covariates=False
    (n_valid_pixels, inds_valid_pixels, valid_values_mask) = valid_data
    (landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy) = data

    n_bands = 6 #landsat_data.shape[0] // (23 * len(years))
    n_output_bands = 6
    n_features = 8

    ds = ArcoV2DatasetV3(None, #Path(f"/mnt/nibble/gen_cog/arcov2/sample_v1.zarr"),  
                         years, sequence_length,
                         limit=10, read_timeless=False,
                         dtype=torch.bfloat16)
    #dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)



    #n_timeless_features = 17 #dataset.n_timeless_features
    model = CfcLearner_v5.load_from_checkpoint(fn_ckpt, map_location='cpu', strict=True) # input_size=input_size, sequence_length=12, output_size=output_size)
    model = model.to(torch.bfloat16)
    model.freeze()

#%%
    pixel_ind = np.random.randint(0, landsat_data.shape[1])

    valid_values = np.nonzero(valid_values_mask[:, pixel_ind])[0]
    #nvv = valid_values.sum()
    #nts = nvv - sequence_length
    first_ind = valid_values[11]+1  # sljedeći datum nakon 12. validne vrijednosti
    next_valid_pos = 12
    n_dates = ds.days_from_start.shape[0]
    x_ls = []; x_ms=[]; x_gt = []; timespans=[]
    for ind in range(first_ind, n_dates):
        # ind = first_ind
        # need to find 12 valid values
        valid_inds=valid_values[next_valid_pos-12:next_valid_pos]
        y_date = ds.days_from_start[ind]
        x_dfs = ds.days_from_start[valid_inds]
        x_lsdata = np.empty((12,n_bands,), dtype=np.float32)
        for b in range(n_bands):
            x_lsdata[:,b] = landsat_data[b*n_dates:(b+1)*n_dates, pixel_ind][valid_inds]

        x_msdata = modis_data[valid_inds, pixel_ind]
        x_gtemp = geom_temp_doy[ds.ind_doys[valid_inds], pixel_ind]
        ts = np.r_[(x_dfs[1:] - x_dfs[:-1]), y_date-x_dfs[-1]]
        x_ls.append(x_lsdata)
        x_ms.append(x_msdata)
        x_gt.append(x_gtemp)
        timespans.append(ts)

        if next_valid_pos < len(valid_values) and valid_values[next_valid_pos] == ind:
            next_valid_pos += 1

    x = np.concatenate((np.array(x_ls)*0.25, 
                        np.expand_dims(np.array(x_ms)/10000, 2), 
                        np.expand_dims(np.array(x_gt)/100, 2)), 
                        axis=2)
    timespans = np.array(timespans,dtype=np.float32)/366
    x[np.isnan(x)] = 0

    y = np.empty((len(valid_values), n_output_bands), dtype=np.float32)
    for b in range(n_output_bands):
        y[:,b] = landsat_data[b*n_dates +valid_values, pixel_ind] *0.25
    
    y_dates = ds.dates[valid_values]
    prd_dates = ds.dates[first_ind:]
                                           


    # tile_data = dataset[1][0]
    # y, x, _, timespans = tile_data
    # y, x, timespans = y.detach().clone(), x.detach().clone(), timespans.detach().clone()
    # x[:,:,7]  = x[:,:,7]/10000
    # x[:,:,8]  = x[:,:,8]/100

    xx = torch.tensor(x, dtype=torch.bfloat16)#.unsqueeze(1)
    tt = torch.tensor(timespans, dtype = torch.bfloat16)#.unsqueeze(1)
    #batchsize = xx.size(0)
    #x_timeless = torch.tensor(timeless).expand(batchsize, -1)
    y_hat = model(xx, tt).detach()

    y_prd = y_hat[valid_values[12:]-first_ind]
    y_obs = torch.tensor(y[12:])


    for b in range(n_output_bands):
        print(f'''Band {b}: 
              MSE={torch.nn.MSELoss()(y_prd[:,b], y_obs[:,b]).item():<.4f}
              R^2={R2Score()(y_obs[:,b], y_prd[:,b]).item():<.4f}
              ''')

    

    y_prd = y_hat.to(torch.float32).numpy()
    fig, axs = plt.subplots(6, 1, figsize=(20, 30))
    for b in range(6):
        ax = axs[b]
        ax.plot(y_dates, y[:,b],'ro', label='observed')
        ax.plot(prd_dates, y_prd[:,b], 'b.-', label='predicted')
        ax.set_title(f"Band {bands_prefix_out[b]}, pixel {pixel_ind}")
        if b==0: 
            ax.legend()
    plt.show()
#%%
def test_whole_image(debug=False):
#%%
    # https://www.intel.com/content/www/us/en/developer/articles/technical/pytorch-quantization-using-intel-neural-compressor.html
    
    time0=time.time()
    model = ov.compile_model(fn_ckpt.parent/f'{fn_ckpt.stem}_quant.xml')
    utils.ttprint(f'Model loaded in {time.time()-time0:.0f} seconds')
#%%    

    tiles = ['090W_49N', '055W_06S','015E_43N']  #'055W_06S'
    year = 2020

    years = np.arange(2000, 2024)
    sequence_length = 12

    for tile in tiles:
        # tile = tiles[0]
        time0 = time.time()
        (success, error, eta), meta, valid_data, data = cfc_sample.get_tile_data(tile)    

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
        
        (n_valid_pixels, inds_valid_pixels, valid_values_mask) = valid_data
        (landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy) = data

        n_bands = 6 #landsat_data.shape[0] // (23 * len(years))
        n_output_bands = 6
        n_features = 8
        sequence_length = 12

        ds = ArcoV2DatasetV3(None,  years, sequence_length,limit=10, read_timeless=False, dtype=torch.bfloat16)
        time1=time.time()
        utils.ttprint(f'Tile {tile} loaded in {time1-time0:.0f} seconds')
        #dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    #%%
        #time1=time.time()
        
        #n_timeless_features = 17 #dataset.n_timeless_features
        
        '''
        model = CfcLearner_v5.load_from_checkpoint(fn_ckpt, map_location='cpu', strict=True) # input_size=input_size, sequence_length=12, output_size=output_size)
        model = model.to(torch.bfloat16)
        model.freeze()
        model = torch.compile(model, backend='openvino')
        '''

        # submodel.eval()
        # ov_model = None
        

        # submodel = model.model.eval()
        # model = ipex.optimize(submodel, 
        #                   dtype=torch.bfloat16, 
        #                   replace_dropout_with_identity=True,
        #                   #election = True
        #                   )
        # model.compile()

        
        #model.freeze()
        #utils.ttprint(f'Model loaded in {time.time()-time1:.0f} seconds')
    #%%
        # predict whole image for some date
        time1 = time.time()
        
        for month in range(1, 13):
            # month=6
            time2 = time.time()
            date = datetime.datetime(year, month, 15)
            doy = date.timetuple().tm_yday
            day_from_start = (date - datetime.datetime(years[0], 1, 1)).days - 1
            n_pixels = landsat_data.shape[1]
            n_dates = ds.days_from_start.shape[0]

            transform = meta[1]
            dtm = covariate_data[covariate_names.index('dtm'), :]
            geom_temp_doy = get_temperature_for_year(transform, dtm)

            #y = np.empty((n_pixels, n_output_bands), dtype=np.float32)
            timespans = np.empty((n_pixels, sequence_length), dtype=np.float32)
            x = np.empty((n_pixels, sequence_length, n_features), dtype=np.float32)
            valid_pixels_ind = np.ones(n_pixels, dtype=bool)

            @njit(parallel=True, fastmath=True, cache=True)
            def _process_one_image(day_from_start, n_pixels, n_bands, 
                                x, timespans, valid_pixels_ind,
                                days_from_start, 
                                landsat_data, 
                                modis_data,
                                geom_temp_doy,
                                valid_values_mask,
                                sequence_length):
                n_dates = days_from_start.shape[0]
                for pix in prange(n_pixels):
                    valid_values = valid_values_mask[:,pix]           
                    pix_dfs = days_from_start[valid_values]
                    ind = np.nonzero(pix_dfs < day_from_start)[0][-sequence_length:]
                    dfs = pix_dfs[ind]
                    if len(ind) < sequence_length:
                        valid_pixels_ind[pix] = False
                        continue
                    for b in range(n_bands):
                        x[pix, :, b] = landsat_data[b*n_dates:(b+1)*n_dates, pix][valid_values][ind]
                    x[pix, :, 6] = modis_data[valid_values, pix][ind]
                    ind_doys = dfs % 23
                    x[pix, :, 7] = geom_temp_doy[ind_doys, pix]
                    timespans[pix, :-1] = dfs[1:] - dfs[:-1]
                    timespans[pix, -1] = day_from_start - dfs[-1]

            _process_one_image(day_from_start, n_pixels, n_bands,
                            x, timespans, valid_pixels_ind,
                            ds.days_from_start,
                            landsat_data, modis_data, geom_temp_doy,
                            valid_values_mask,
                            sequence_length)


            if not valid_pixels_ind.all():
                x = x[valid_pixels_ind, :, :]
                timespans = timespans[valid_pixels_ind, :]
                #timeless = covariate_data[:, valid_pixels_ind].T
            #else:
            #    timeless = covariate_data.T

            x[np.isnan(x)] = 0
            x[:,:, :6] = x[:,:, :6] * 0.25
            x[:,:,6]  = x[:,:,6]/10000
            x[:,:,7]  = x[:,:,7]/100

    #%%
            # xx = torch.tensor(x, dtype=torch.bfloat16) #.unsqueeze(1)
            # tt = torch.tensor(timespans, dtype=torch.bfloat16) #.unsqueeze(1)

            xx = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
            tt = torch.tensor(timespans, dtype=torch.float32).unsqueeze(1)
            #x_timeless = torch.tensor(timeless).expand(batchsize, -1)

            # y_bf = model(torch.tensor(x, dtype=torch.bfloat16), torch.tensor(timespans, dtype=torch.bfloat16))[0]

            del x, timespans

            # n=xx.shape[0]//2
            # executor = ThreadPoolExecutor(max_workers=2)            
            # f1 = executor.submit(model, (xx[:n,:,:,:], tt[:n,:,:]))
            # f2 = executor.submit(model, (xx[n:,:,:,:], tt[n:,:,:]))
            # y_hat = []
            # for f in [f1, f2]:
            #     res = f.result()[0]
            #     #y_hat = y_hat.detach().numpy()
            #     np.clip(res, 0, 1, out=res)
            #     y_hat.append(res)


           
            #y_hat = np.concatenate(y_hat, axis=0)
            y_hat = model((xx, tt))[0]
            np.clip(y_hat, 0, 1, out=y_hat)
            del xx, tt
            gc.collect()



    # 8min for full image
# %%
            y_hat = (y_hat*40000).astype(np.uint16)
            prd = np.full(n_pixels, nodata, dtype=np.uint16)
            for b in range(n_output_bands):
                # b=0
                band_name = bands_prefix_out[b]
                fn = fld_out / f"{tile}_{year}{month:02d}_{band_name}.tif"
                if not valid_pixels_ind.all():
                    prd[:] = nodata
                    prd[valid_pixels_ind] = y_hat[:, b]                    
                else:
                    prd = y_hat[:, b]

                # vmin, vmax = np.percentile(prd, [2, 98])
                # plt.imshow(prd.reshape(utils.y_size, utils.x_size), cmap='YlGn', vmin=vmin, vmax=vmax)
                # plt.title(f"{tile} {year}-{month:02d} {band_name}")
                # plt.show()

                with rasterio.open(fn, 'w', **profile) as dst:
                    dst.write(prd.reshape(utils.y_size, utils.x_size), 1)

            utils.ttprint(f"Month {month}. processed in {(time.time()-time2)/60:.2f} minutes")

        del landsat_data, modis_data, covariate_data, covariate_names, geom_temp_doy

        utils.ttprint(f"Whole year processed in {(time.time()-time1)/60:.2f} minutes")
        utils.ttprint(f"Total time for {tile} is {(time.time()-time0)/60:.2f} minutes")
        del y_hat, prd
        gc.collect()

# %% Create GIF
def create_gif(dfc, fn_gif):
    vmin = None; vmax=None
    data = []
    for fn in dfc['fn']:
        with rasterio.open(fn) as src:
            data.append(src.read(1,masked=True))
            # Create GIF for each band
            #create_gif(data, band, year)
    data = np.array(data)
    vmin, vmax = np.percentile(data, [2, 98])

    # system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    myFont = ImageFont.truetype('/root/.local/share/mamba/envs/arcov2/fonts/SourceCodePro-Regular.ttf', 65)
    imgs = []
    for i in range(data.shape[0]):
        row = dfc.iloc[i]
        # img = data[i]
        # img = (img - vmin) / (vmax - vmin)
        # img = Image.fromarray(np.uint8(matplotlib.cm.gist_earth(img)*255)).resize((2002,2002), Image.Resampling.LANCZOS)
        # ID = ImageDraw.Draw(img)
        # 
        # ID.text((10,10), f"{row.tile} Band: {row.band}, {row.year}-{row.month:02d}", font=myFont, fill=(255,0,0))
        # img = img.resize((2002,2002))
        plt.close('all')
        fig, ax = plt.subplots(figsize=(8,8), dpi=100)
        ax.imshow(data[i], cmap='gist_earth', vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{row.tile} {row.band} {row.year}-{row.month:02d}")
        fig.tight_layout(pad=1)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        imgs.append(img)

    frame_one = imgs[0]
    frame_one.save(fn_gif, format="GIF", append_images=imgs,
               save_all=True, duration=1000, loop=0)

    # Create GIF
    #imageio.mimsave(f"gif_{dfc['band'].values[0]}_{dfc['year'].values[0]}.gif", imgs, duration=0.5)


def create_all_gifs():
    fns = fld_out.glob("*.tif")
    data = []
    for fn in fns:
        ss = fn.stem.split('_')
        data.append(dict(tile=ss[0]+'_'+ss[1], year=ss[2][:4], month=int(ss[2][4:6]), band=ss[3], fn=fn))
    df = pandas.DataFrame(data)

    tiles = df['tile'].unique()
    for tile in tiles:
        tile_df = df[df['tile'] == tile]
        years = tile_df['year'].unique()
        bands = sorted(tile_df['band'].unique())
        for year in years:
            for band in bands:
                fn_gif = fld_gifs / f"{tile}_{year}_{band}.gif"                
                if not fn_gif.exists():
                    utils.ttprint(f"Creating {fn_gif}")
                    dfc = tile_df[(tile_df['band'] == band) & (tile_df['year'] == year)].sort_values('month')
                    create_gif(dfc, fn_gif)

                # band_name = bands_prefix_out[b]
                #  plt.imshow(prd.reshape(utils.y_size, utils.x_size), cmap='YlGn')
                #  plt.gca().set_xticks([])  # Remove x-axis ticks
                #  plt.gca().set_yticks([])  # Remove y-axis ticks
                #  plt.title(band_name)
                #  plt.show()

# %% 0,3,2
    #plt.imshow(y_hat[:, [0,3,2]].reshape(utils.y_size, utils.x_size, 3)/1.272)
# %%
if __name__ == "__main__":
    #statistics()
    test_whole_image()
    #create_all_gifs()


# Timings
'''
[11:30:52] Loading tile 055W_06S
[11:32:51] Landsat data loaded in 118.92 seconds
[11:34:29] Processing 552 MODIS NDVI files in parallel...
[11:40:42] MODIS NDVI data loaded in 471.31 seconds
[11:40:42] Landsat + modis: 590.24 seconds
[11:40:42] Masking data ...
[11:41:18] Masked Landsat data from QA in 36.31 seconds
[11:44:49] Masked Landsat data from MODIS in 210.94 seconds
[11:44:49] Masked Landsat data in 247.25 seconds
[11:44:49] Scaling and trimming Landsat data ...
[11:45:03] Scaled Landsat data by 10000
[11:45:04] Trimmed Landsat data to 3864 rows
[11:45:04] Landsat data scaled and trimmed in 14.47 seconds
[11:45:22] Getting covariates ...
[11:47:49] Covariates loaded in 146.15 seconds
[11:47:49] Getting geom_temp_doy ...
[11:47:50] Got geom_temp_doy in 1.50 seconds
[11:47:50] Total time for loading tile 055W_06S: 1018.36 seconds
[11:47:50] Tile 055W_06S loaded in 1018 seconds
[11:47:51] Model loaded in 0 seconds
[11:53:46] Month 1. processed in 5.92 minutes
[11:59:45] Month 2. processed in 5.98 minutes
[12:05:48] Month 3. processed in 6.06 minutes
[12:11:53] Month 4. processed in 6.08 minutes
[12:17:52] Month 5. processed in 5.98 minutes
[12:23:50] Month 6. processed in 5.96 minutes
[12:29:49] Month 7. processed in 5.97 minutes
[12:35:47] Month 8. processed in 5.97 minutes
[12:41:46] Month 9. processed in 5.98 minutes
[12:47:46] Month 10. processed in 5.99 minutes
[12:53:46] Month 11. processed in 6.01 minutes
[12:59:45] Month 12. processed in 5.98 minutes
[12:59:45] Whole year processed in 71.90 minutes
[12:59:45] Total time for 055W_06S is 88.88 minutes

######################################
with ipex:
[15:25:45] Tile 055W_06S loaded in 257 seconds
[15:25:45] Model loaded in 0 seconds
[15:30:52] Month 1. processed in 5.11 minutes
[15:35:37] Month 2. processed in 4.75 minutes
[15:40:22] Month 3. processed in 4.76 minutes
[15:45:11] Month 4. processed in 4.81 minutes
[15:50:02] Month 5. processed in 4.85 minutes
[15:54:51] Month 6. processed in 4.82 minutes
[15:59:38] Month 7. processed in 4.79 minutes
[16:04:28] Month 8. processed in 4.83 minutes
[16:09:17] Month 9. processed in 4.81 minutes
[16:14:05] Month 10. processed in 4.81 minutes
[16:18:51] Month 11. processed in 4.76 minutes
[16:23:38] Month 12. processed in 4.79 minutes
[16:23:38] Whole year processed in 57.89 minutes
[16:23:38] Total time for 055W_06S is 62.18 minutes

'''