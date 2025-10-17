#%%
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from re import A
from typing import Any
from zipfile import ZipFile
from numba import njit, prange
import numpy as np
from numpy.typing import NDArray
import pympler.asizeof
import torch
from torch.utils.data import Dataset,DataLoader
from pathlib import Path    
import zarr
from datetime import datetime, timedelta
import time
import pympler
from tqdm import tqdm

from settings import n_imag_per_year,n_threads
from utils import ttprint
#%%
