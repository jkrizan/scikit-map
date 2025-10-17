# Notes about v1
- used maskQA (without buffer), agregate and then SWA

# Notes about v2

- consider, 1 minute of computing time per tile corresponds to 1 day in production, this means that if you stay below 15-20 minutes it's ok
- landsat_data - rows = n_years * n_imag_per_year * (n_spect_bands + 2), extra bands are NDVI and QA

- SWA should be replaced with some model
- All pixels should be predicted (smoothed) through timeseries and result recorded for every month on 15. date
- Remove all snow
    - should be alredy removed via `mask_from_qa`
- Keep water 
- Sugestion: use spline with few params
- Suggestion: use lasso 
- Suggestion: use this variables:
    - LU/LC - 30m, WGS84
    http://192.168.1.30:8333/global/lc/lc_glad.glcluc_c_30m_s_20150101_20151231_go_epsg.4326_v2.tif
    http://192.168.1.30:8333/global/lc/lc_glad.glcluc_c_30m_s_20160101_20161231_go_epsg.4326_v2.tif
    http://192.168.1.30:8333/global/lc/lc_glad.glcluc_c_30m_s_20170101_20171231_go_epsg.4326_v2.tif
    http://192.168.1.30:8333/global/lc/lc_glad.glcluc_c_30m_s_20180101_20181231_go_epsg.4326_v2.tif
    http://192.168.1.30:8333/global/lc/lc_glad.glcluc_c_30m_s_20190101_20191231_go_epsg.4326_v2.tif
    http://192.168.1.30:8333/global/lc/lc_glad.glcluc_c_30m_s_20200101_20201231_go_epsg.4326_v2.tif
    http://192.168.1.30:8333/global/lc/lc_glad.glcluc_c_30m_s_20210101_20211231_go_epsg.4326_v2.tif
    http://192.168.1.30:8333/global/lc/lc_glad.glcluc_c_30m_s_20220101_20221231_go_epsg.4326_v2.tif
    http://192.168.1.30:8333/global/lc/lc_glad.glcluc_c_30m_s_20230101_20231231_go_epsg.4326_v2.tif

    - DTM
        - hillshade (but don+t have it)
        - slope
        - negative openness
        - base_path is any http://192.168.49.X:8333 with X from 30 to 46. If you read in parallel is good to use different X for different files to distribute the requests 
        {base_path}/global/edtm/legendtm_rf_30m_m_s_20000101_20231231_go_epsg.4326_v20250130.tif
        {base_path}/global/dtm/pos.openness_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20240528.tif
        {base_path}/global/dtm/neg.openness_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20240528.tif
        {base_path}/global/dtm/slope_edtm_m_30m_s_20000101_20221231_go_epsg.4326_v20240528.tif
        {base_path}/global/dtm/v3/dfme_edtm_m_120m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/dfme_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/dfme_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/geomorphon_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/geomorphon_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/hillshade_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/ls.factor_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/ls.factor_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/maxic_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/maxic_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/minic_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/minic_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/neg.openness_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/neg.openness_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/pos.openness_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/pos.openness_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/pro.curv_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/pro.curv_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/ring.curv_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/ring.curv_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/shpindx_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/shpindx_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/slope.in.degree_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/spec.catch_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/ssdon_edtm_m_120m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/ssdon_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/tan.curv_edtm_m_120m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/tan.curv_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/tan.curv_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/twi_edtm_m_120m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/twi_edtm_m_240m_s_20000101_20221231_go_epsg.4326_v20241230.tif
        {base_path}/global/dtm/v3/twi_edtm_m_60m_s_20000101_20221231_go_epsg.4326_v20241230.tif

- Should be done in less then 15 minuts per tile without IO
- Articles to consider:
- https://www.mdpi.com/1424-8220/25/5/1622
- https://doi.org/10.5194/essd-16-5449-2024 
- https://doi.org/10.3390/s25051622
- https://doi.org/10.1016/j.isprsjprs.2021.08.015 
- https://doi.org/10.3390/rs13030484
- https://doi.org/10.3390/ijgi12060214
- https://essd.copernicus.org/articles/16/5449/2024/

GLAD dataset is on this location on web: 
https://glad.umd.edu/dataset/glad_ard2/33N/000E_33N/921.tif
https://glad.umd.edu/dataset/glad_ard2/{tile1}/{tile0}_{tile1}/{interval_id}.tif
But, for every image there is a txt file with name "{interval_id}_{n_originals}.txt, where `n_originals` is number of landsat images used for this composite and in text file there are list of that images ...
For example in "33N/000E_33N/924_12.txt":
    ```
    198036_2020053_LC08
    198037_2020053_LC08
    197036_2020054_LE07
    197037_2020054_LE07
    196037_2020055_LC08
    196036_2020055_LC08
    198036_2020061_LE07
    198037_2020061_LE07
    197037_2020062_LC08
    197036_2020062_LC08
    196036_2020063_LE07
    196037_2020063_LE07
    ```

### Loading of data for 1 tile
- Some tiles are empty for landsat_data, and have modis_data 094W_79N, 134E_54N

## CfC
class CfC(nn.Module)
    - def forward(self, x, timespans=None, mask=None):
    - x has 3 dimensions
        - 0 - batch size
        - 1 - sequence length
        - 2 - features
    - Think this is error at torch_cfc.py:193 (for t in range(seq_len):  inputs = x[:, t]) ... maybe not ...
## Ideas for v3
1. Sample some number of pixels from image where LS data is not nan for all dates (maybe avoid to sample pixels that have less valid dates)
2. Make model that have target variables all 7 spectral bands from landsat, and x variables are
    - modis ndvi
    - DOY (or maybe to convert it to winterness and summerness)
    - year (as nominal variable)
    - DTM variables (height, slope, negative opennes ... maybe all that I have)
    - LULC (as nominal, maybe best class2 level - class1 has 6 non-ocean classes, class2 has 12 non-ocean classes)
3. Inference is done for every month - just changing year and DOY to 15. day in that month


## Some things about PyTorch
### How to split train/validation
# Initialize KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Split the dataset into 5 folds
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    print(f"Fold {fold}:")
    print(f"Train indices: {train_idx[:5]}")
    print(f"Validation indices: {val_idx[:5]}")

    # Create subsets for training and validation
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    print(f"Train subset size: {len(train_subset)}, Validation subset size: {len(val_subset)}")

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=2)


v6 -
- MAE: 0.025632629171013832
  - MSE: 0.0012307808501645923
  - R2: 0.9173867702484131