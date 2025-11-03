#%%
from numpy.random import f
from settings import MASTER_LOGGER_FILE, LOG_FOLDER, TILES_FILE
import pandas
from datetime import datetime, timedelta
import numpy as np

#%%
def master_logger_monitoring() -> None:
#%%
    quotechar = r'`'
    #print(ord(quotechar))
    df = pandas.read_table(MASTER_LOGGER_FILE, sep=r",",quotechar=quotechar, quoting=1, header=0, engine='python', 
                           skipinitialspace=True, parse_dates=['timestamp'])    
    #df.head()

    started_tiles = df.query("event == 'PRODUCE_TILE' and status=='START'")['message'].str.extract(r'TILE=([\d+\w+-]+)')[0].tolist()
    finished_tiles = df.query("event == 'PRODUCE_TILE' and status=='SUCCESS'")['message'].str.extract(r'TILE=([\d+\w+-]+)')[0].tolist()

    print(f"Started tiles: {len(started_tiles)}")
    print(f"Finished tiles: {len(finished_tiles)}")

    failed_tiles = set(started_tiles) - set(finished_tiles)
    no_valid_pixels=[]
    still_running_tiles = []
    for tile in failed_tiles:
        # tile = list(failed_tiles)[0]; tile='098E_52N'
        fn = LOG_FOLDER / f"{tile}.log"
        dft = pandas.read_table(fn, sep=r",",quotechar=quotechar, quoting=1, header=0, engine='python', skipinitialspace=True)
        failed_step = dft.query("status=='FAILED'")['message'].tolist()
        if len(failed_step) == 0:
            msg = f'TILE={tile}'
            dff = df.query("message==@msg and status=='START' and event=='PRODUCE_TILE'")[['hostname','ip_address', 'timestamp']]
            still_running_tiles.append((tile, f'{msg}, still running for {(datetime.now() - dff.iloc[0]["timestamp"]).seconds // 60} minutes on {dff.iloc[0]["hostname"]} ({dff.iloc[0]["ip_address"]}) since {dff.iloc[0]["timestamp"]}'))
        else:
            message = failed_step[0]
            if message.startswith("No valid pixels for tile"):
                no_valid_pixels.append(tile)
            else:
                print(f"Tile {tile} failed at step: {message}")
   
    if len(no_valid_pixels) > 0:
        print(f"Tiles with no valid pixels: {len(no_valid_pixels)}")

    if len(still_running_tiles) > 0:
        print("Tiles still running:")
        for tile, info in still_running_tiles:
            print(f"  {info}")

    timings = []
    for tile in finished_tiles:
        # tile = finished_tiles[0]; tile='098E_52N'
        msg = f'TILE={tile}'
        dff = df.query("message==@msg and status=='SUCCESS' and event=='PRODUCE_TILE'")
        time_taken = dff['duration'].iloc[0]
        timings.append(time_taken / 60.0)
    if len(timings) > 0:
        print(f"Average time per finished tile: {np.mean(timings):.2f} minutes")
        print(f"Standard deviation of time per finished tile: {np.std(timings):.2f} minutes")
        print(f"Median time per finished tile: {np.median(timings):.2f} minutes")
        print(f"Max time per finished tile: {np.max(timings):.2f} minutes")
        print(f"Min time per finished tile: {np.min(timings):.2f} minutes")
     
# %%
