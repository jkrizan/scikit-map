#%%
import gc
import json
import time

import data_utils as dutils
import production_utils as putils
from settings import DATES_TO_PREDICT, MODEL_OPTIMIZATION, N_NEEDED_VALID_DATES, PREDICTIONS_FOLDER, SETTINGS_DICT, FILE_ENDING_OUT, MODEL_NAME


# %%
def production():
    putils.init()

    master_logger = putils.get_master_logger()
    master_logger.log("INIT_PRODUCTION", "START", 0, json.dumps(SETTINGS_DICT))

    time0 = time.time()
    model = putils.load_model()
    model = putils.optimize_model(model)
    master_logger.log(
        "LOAD_MODEL",
        "SUCCESS",
        time.time() - time0,
        f"Model {MODEL_NAME} loaded and optimized using {MODEL_OPTIMIZATION}",
    )  
    
    try:
        while True:
            tile, log = putils.choose_landsat_tile()
            if tile is None or log is None:
                print("All tiles are processed.")
                break

            start_time = time.time()
            master_logger.log("PRODUCE_TILE", "START", 0, f"TILE={tile}")
            log.log(
                "START_PRODUCTION", "START", 0, f"Starting production for tile {tile}"
            )

            data = dutils.load_tile_data(tile, log)
            profile = data.pop("profile")

            time0 = time.time()
            n_valid_pixels = data['mask_valid_pixels'].sum()
            if n_valid_pixels == 0:
                log.log(
                    "DATA_VALIDATION",
                    "FAILED",
                    time.time() - time0,
                    f"No valid pixels for tile {tile}",
                )
                continue
            log.log(
                "DATA_VALIDATION",
                "SUCCESS",
                time.time() - time0,
                f"{n_valid_pixels}",
            )

            OUTPUT_FOLDER = PREDICTIONS_FOLDER / tile
            OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

            for predict_date in DATES_TO_PREDICT:
                # predict_date = DATES_TO_PREDICT[0]
                gc.collect()

                fn_output = (
                    OUTPUT_FOLDER
                    / f"{tile}_{predict_date.year}{predict_date.month:02d}{FILE_ENDING_OUT}.tif"
                )
                if fn_output.exists():
                    log.log(
                        "PREDICT_DATE",
                        "SKIPPED",
                        0,
                        f"Predictions for {tile} {predict_date.year}-{predict_date.month:02d} already exist, skipping",
                    )
                    continue
                else:
                    log.log("PREDICT_DATE", "START", 0, fn_output.name)

                time0 = time.time()
                prepared_data = dutils.prepare_data_for_date(data, predict_date)
                count_pixels = prepared_data[-1].sum()
                if count_pixels == 0:
                    log.log(
                        "DATA_PREPARATION",
                        "FAILED",
                        time.time() - time0,
                        f"No valid pixels for date {predict_date.year}-{predict_date.month:02d}",
                    )
                    continue
                log_dict = dict(date=predict_date.strftime("%Y-%m-%d"), count_pixels=int(count_pixels))
                log.log(
                    "DATA_PREPARATION",
                    "SUCCESS",
                    time.time() - time0,
                    json.dumps(log_dict),
                )

                time0 = time.time()
                predictions = putils.run_inference(model, prepared_data)
                log.log(
                    "INFERENCE",
                    "SUCCESS",
                    time.time() - time0,
                    f"Inference completed for month {predict_date.year}-{predict_date.month:02d}",
                )

                time0 = time.time()
                dutils.save_predictions(predictions, profile, prepared_data[-1], fn_output)
                log.log(
                    "IMAGE_SAVING",
                    "SUCCESS",
                    time.time() - time0,
                    f"Predictions saved for month {predict_date.year}-{predict_date.month:02d}",
                )

            master_logger.log(
                "PRODUCE_TILE", "SUCCESS", time.time() - start_time, f"TILE={tile}"
            )

    finally:
        putils.clean()

if __name__ == "__main__":
    production()