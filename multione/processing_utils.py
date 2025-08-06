

from numpy.typing import NDArray, ArrayLike
from typing import Any, Tuple, Union

import numpy as np


def get_SWA_weights(att_env, att_seas, season_size, n_imag) -> NDArray:
    # TODO: float32
    conv_mat_row = np.zeros((n_imag))
    base_func = np.zeros((season_size,))
    period_y = season_size/2.0
    slope_y = att_seas/10/period_y
    for i in np.arange(season_size):
        if i <= period_y:
            base_func[i] = -slope_y*i
        else:
            base_func[i] = slope_y*(i-period_y)-att_seas/10
    # Compute the envelop to attenuate temporarly far images
    env_func = np.zeros((n_imag,))
    delta_e = n_imag
    slope_e = att_env/10/delta_e
    for i in np.arange(delta_e):
        env_func[i] = -slope_e*i
        conv_mat_row = 10.0**(np.resize(base_func,n_imag) + env_func)
    return conv_mat_row


def process_image_in_chunks(image, chunk_size, gap_stripes_th, gap_general_th, fft_th):
    mask = np.isnan(image)
    height, width = image.shape
    n_chunk_height = int(np.floor(height/chunk_size))
    n_chunk_width = int(np.floor(width/chunk_size))
    gap_fraq = np.zeros((n_chunk_height, n_chunk_width))
    fft_score = np.zeros((n_chunk_height, n_chunk_width))
    rec_flag = np.zeros((n_chunk_height, n_chunk_width))
    #output_image = image.copy()
    row_starts, row_ends, col_starts, col_ends, fill_true_erase_false = [], [], [], [], []
    
    # Loop through the image by chunks
    for i in range(0, n_chunk_height):
        for j in range(0, n_chunk_width):
            # @FIXME check is also theretically the location of patial frequencies in different share chunks is the same 
            if i != (n_chunk_height-1):
                row_start, row_end = (i * chunk_size, (i+1) * chunk_size)
            else:
                row_start, row_end = (i * chunk_size, height)
            if j != (n_chunk_width-1):
                col_start, col_end = (j * chunk_size, (j+1) * chunk_size)
            else:
                col_start, col_end = (j * chunk_size, width)
            image_chunk = image[row_start:row_end, col_start:col_end]
            mask_chunk = mask[row_start:row_end, col_start:col_end]
            gap_count_chunk = np.sum(mask_chunk)
            gap_fraq[i, j] = gap_count_chunk/(row_end-row_start)/(col_end-col_start)
            if gap_fraq[i, j] < gap_general_th:
                row_starts += [row_start]
                row_ends += [row_end]
                col_starts += [col_start]
                col_ends += [col_end]
                fill_true_erase_false += [True]
                rec_flag[i,j] = 1
            else:
                image_filled = np.nan_to_num(image_chunk, nan=0)
                image_filled = image_filled[0:chunk_size,0:chunk_size].copy()
                # image_filled /= max(np.max(image_filled),1)
                image_filled[image_filled!=0] = 1
                ft = np.fft.ifftshift(image_filled)
                ft = np.fft.fft2(ft, norm='ortho')
                ft = np.fft.fftshift(ft)
                ft[48:80,48:80] = 0
                fft_score[i, j] = np.max(np.abs(ft))
                if fft_score[i, j] > fft_th:
                    row_starts += [row_start]
                    row_ends += [row_end]
                    col_starts += [col_start]
                    col_ends += [col_end]
                    if gap_fraq[i, j] < gap_stripes_th:
                        fill_true_erase_false += [True]
                        rec_flag[i,j] = 1
                    else:
                        fill_true_erase_false += [False]
                        rec_flag[i,j] = -1
                    
    return row_starts, row_ends, col_starts, col_ends, fill_true_erase_false, gap_fraq, fft_score, rec_flag

#%%

def HANTS(ni, nb, nf, y, ts, HiLo, low, high, fet, dod, delta, fill_val):
    '''
    This function applies the Harmonic ANalysis of Time Series (HANTS)
    algorithm originally developed by the Netherlands Aerospace Centre (NLR)
    (http://www.nlr.org/space/earth-observation/).

    This python implementation was based on two previous implementations
    available at the following links:
    https://codereview.stackexchange.com/questions/71489/harmonic-analysis-of-time-series-applied-to-arrays
    http://nl.mathworks.com/matlabcentral/fileexchange/38841-matlab-implementation-of-harmonic-analysis-of-time-series--hants-
    '''

    """
    ni    = nr. of images (total number of actual samples of the time series)
    nb    = length of the base period, measured in virtual samples 
            (days, dekads, months, etc.)        
    nf    = number of frequencies to be considered above the zero frequency
    y     = array of input sample values (e.g. NDVI values)
    ts    = array of size ni of time sample indicators 
            (indicates virtual sample number relative to the base period); 
            numbers in array ts maybe greater than nb
            If no aux file is used (no time samples), we assume ts(i)= i, 
            where i=1, ..., ni         
    HiLo  = 2-character string indicating rejection of high or low outliers
            select from 'Hi', 'Lo' or 'None'    
    low   = valid range minimum
    high  = valid range maximum (values outside the valid range are rejeced
            right away)    
    fet   = fit error tolerance (points deviating more than fet from curve 
            fit are rejected)
    dod   = degree of overdeterminedness (iteration stops if number of 
            points reaches the minimum required for curve fitting, plus 
            dod). This is a safety measure            
    delta = small positive number (e.g. 0.1) to suppress high amplitudes                  
    """

    '''
    ni=len(ts_modis); nb=365; nf=5; y=ts_modis; ts=t; HiLo='Hi'; low=0.1; high=0.9; fet=0.1; dod=0; delta=0.1; fill_val=np.nan
    '''
    from copy import deepcopy


    # Arrays
    mat = np.zeros((min(2*nf+1, ni), ni), dtype=np.float32)  # type: ignore
    # amp = np.zeros((nf + 1, 1))
    # phi = np.zeros((nf+1, 1))
    yr = np.zeros((ni, 1), dtype=np.float32)  # type: ignore
    y_len = len(y)
    outliers = np.zeros((1, y_len), dtype=np.float32)  # type: ignore

    # Filter
    sHiLo = 0
    if HiLo == 'Hi':
        sHiLo = -1
    elif HiLo == 'Lo':
        sHiLo = 1

    nr = min(2*nf+1, ni)
    noutmax = ni - nr - dod
    # dg = 180.0/math.pi
    mat[0, :] = 1.0

    ang = 2*np.pi * np.arange(nb)/nb
    cs = np.cos(ang)
    sn = np.sin(ang)

    i = np.arange(1, nf+1)
    for j in range(ni):
        index = np.mod(i*ts[j], nb)
        mat[2 * i-1, j] = cs.take(index)
        mat[2 * i, j] = sn.take(index)

    p = np.ones_like(y)
    bool_out = (y < low) | (y > high)
    p[bool_out] = 0
    outliers[bool_out.reshape(1, y.shape[0])] = 1
    nout = np.sum(p == 0)

    if nout > noutmax:
        if np.isclose(y, fill_val).any():
            ready = np.array([True])
            yr = y
            outliers = np.zeros((y.shape[0]), dtype=int)
            outliers[:] = fill_val
        else:
            raise Exception('Not enough data points.')
    else:
        ready = np.zeros((y.shape[0]), dtype=bool)

    nloop = 0
    nloopmax = ni

    while ((not ready.all()) & (nloop < nloopmax)):

        nloop += 1
        za = np.matmul(mat, p*y)

        A = np.matmul(np.matmul(mat, np.diag(p)), np.transpose(mat))
        A = A + np.identity(nr)*delta
        A[0, 0] = A[0, 0] - delta

        zr = np.linalg.solve(A, za)

        yr = np.matmul(np.transpose(mat), zr)
        diffVec = sHiLo*(yr-y)
        err = p*diffVec

        err_ls = list(err)
        #err_sort = deepcopy(err)
        #err_sort.sort()
        err.sort()

        rankVec = [err_ls.index(f) for f in err]

        maxerr = diffVec[rankVec[-1]]
        ready = (maxerr <= fet) | (nout == noutmax)

        if (not ready):
            i = ni - 1
            j = rankVec[i]
            while ((p[j]*diffVec[j] > 0.5*maxerr) & (nout < noutmax)):
                p[j] = 0
                outliers[0, j] = 1
                nout += 1
                i -= 1
                if i == 0:
                    j = 0
                else:
                    j = 1

    return [yr, outliers]

