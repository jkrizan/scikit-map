#%%
from openvino.runtime import Core, properties   # Here is warning:"Numba: Attempted to fork from a non-main thread, the TBB library may be in an invalid state in the child process."
import torch.onnx
import ray
import json
from settings import DATA_TYPE, DEVICE, PRODUCTION_FOLDER, MODEL_NAME, MODEL_INPUT_SIZE, MODEL_OUTPUT_SIZE, MODEL_TIMELESS_SIZE, MODEL_SEQUENCE_LENGTH, RND_GEN
from settings import MODEL_SUBFOLDER, DATASET_ZARR, YEARS
from settings import N_THREADS_INFERENCE, TILES_FILE, PRODUCTION_BATCH_SIZE, MODEL_OPTIMIZATION
from settings import LOG_FOLDER, LOGGER_SILENT, MASTER_LOGGER_FILE, MODEL_PREDICTION_MIN, MODEL_PREDICTION_MAX
from production_logger import ProductionLogger
import torch
from filelock import FileLock
import data_utils
from cfc import CfcModel, ArcoV2Dataset
from typing import Any
import numpy as np
from io import BytesIO

#%%

def get_master_logger() -> ProductionLogger:
    lock = FileLock(MASTER_LOGGER_FILE.with_suffix('.lock'))
    lock.acquire()
    try:        
        logger = ProductionLogger(MASTER_LOGGER_FILE, silent=True, master=True, hostname=get_host_name(), ip_address=get_ip_address())
    finally:
        lock.release()
    return logger

def get_host_name() -> str:
    # Other posible implementations:
    # docker run -it -e "HOSTNAME=$(cat /etc/hostname)" <image> <cmd>
    # and then read os.environ['HOSTNAME']
    import socket
    return socket.gethostname()

def get_ip_address() -> str:
    # --network=host
    import socket
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

def init():
    ray.init(ignore_reinit_error=True)

def clean():
    ray.shutdown()


def choose_landsat_tile()-> tuple[str | None, data_utils.ProductionLogger | None]:
    tiles_all = open(TILES_FILE, 'r').read().splitlines()
    tiles_done = [f.stem for f in LOG_FOLDER.glob('*.log')]
    #tiles_togo = list(set(tiles_all) - set(tiles_done))
    for tile in tiles_all:
        if tile in tiles_done:
            continue
        log_file = LOG_FOLDER / f"{tile}.log"
        if not log_file.exists():
            lock = FileLock(LOG_FOLDER / f"{tile}.lock")
            lock.acquire()
            try:
                if not log_file.exists():
                    log = data_utils.ProductionLogger(log_file, silent=LOGGER_SILENT)
                    return tile, log                                
            finally:
                lock.release()

    return None, None

def load_dataset() -> ArcoV2Dataset:

    dataset = ArcoV2Dataset(
        zarr_path=DATASET_ZARR,
        years=YEARS,
        sequence_length=MODEL_SEQUENCE_LENGTH,
        indices=['fpar'],
        limit=(190,200),
        percent_pixels=0.5,
        device='cpu',
        dtype=torch.float32,
        return_tensors=True        
    )
    dataset.prepare_all_cases()

    return dataset

def load_model() -> CfcModel:

    fld_model = PRODUCTION_FOLDER / MODEL_SUBFOLDER
    config = json.load(open(fld_model / f"{MODEL_NAME}.json", "r"))['train_loop_config']

    model = CfcModel(
        input_size=MODEL_INPUT_SIZE,
        output_size=MODEL_OUTPUT_SIZE,
        timeless_input_size=MODEL_TIMELESS_SIZE,
        sequence_length=MODEL_SEQUENCE_LENGTH,
        backbone_layers=[config["n_backbone_size"]] * config["n_backbone_layers"],
        hidden_size=config["hidden_size"],
    )

    ray_state_dict = torch.load(fld_model / f"{MODEL_NAME}.pt", map_location='cpu')[0]
    state_dict = {k.replace('module.', ''): v for k, v in ray_state_dict.items()}
    model.load_state_dict(state_dict)

    return model

def optimize_model(model: CfcModel) -> Any:
    if MODEL_OPTIMIZATION == "OPENVINO":
        optimized_model = optimize_model_openvino(model, load_dataset())
    else:
        raise ValueError(f"Unknown MODEL_OPTIMIZATION: {MODEL_OPTIMIZATION}")
    return optimized_model

class OpenVINOModelWrapper(torch.nn.Module):
        def __init__(self, compiled_model):
            super(OpenVINOModelWrapper, self).__init__()
            self.compiled_model = compiled_model

        def forward(self, x, tl, ts):
            inputs = {
                'input_x': x, #.numpy(),
                'input_tl': tl, #.numpy(),
                'input_ts': ts #.numpy()
            }
            result = self.compiled_model.infer_new_request(inputs)
            return result['output']    

def optimize_model_openvino(model: CfcModel, dataset: ArcoV2Dataset) -> OpenVINOModelWrapper:
    # dataset = load_dataset()
    
    model.forward = model.inference  # Use inference method for optimization

    # Export to ONNX
    _, dummy_input_x, dummy_input_tl, dummy_input_ts = dataset.get_cases(RND_GEN.choice(len(dataset), size=1024, replace=False))    
    dummy_input_ts = dummy_input_ts[:, 1:]  # Remove one timespan to match model input for inference
    inds = RND_GEN.choice(dummy_input_x.shape[0], size=512, replace=False)
    dummy_input_ts[inds] = - dummy_input_ts[inds]  # Introduce some backward series

    #dummy_input = torch.randn(1, MODEL_SEQUENCE_LENGTH, MODEL_INPUT_SIZE)
    #dummy_timeless = torch.randn(1, MODEL_TIMELESS_SIZE)
    #onnx_path = PRODUCTION_FOLDER / MODEL_SUBFOLDER / f"{MODEL_NAME}.onnx"
    onnx_object = BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_x, dummy_input_tl, dummy_input_ts),
            onnx_object,     # type: ignore
            input_names=['input_x', 'input_tl', 'input_ts'],
            output_names=['output'],
            dynamic_axes={'input_x': {0: 'batch_size'},
                        'input_tl': {0: 'batch_size'},
                        'input_ts': {0: 'batch_size'},
                        'output': {0: 'batch_size'}},
            # dynamo=True,        ??? To try
        )
    onnx_object.seek(0)
    # Load and optimize with OpenVINO
    core = Core()
    ov_model = core.read_model(model=onnx_object)
    compile_config = {  # Best practices for CPU performance in comments
        properties.inference_num_threads(): N_THREADS_INFERENCE,
        properties.hint.enable_hyper_threading(): False,    # False
        properties.hint.enable_cpu_pinning(): True, # True
        properties.hint.performance_mode(): properties.hint.PerformanceMode.LATENCY, #LATENCY
        # the value of ov::num_streams is calculated by dividing ov::inference_num_threads by the number of threads per stream.
        # properties.num_streams(): mp.cpu_count() // 2,  # assuming 4 threads per stream
    }    
    compiled_model = core.compile_model(ov_model, device_name=DEVICE, config=compile_config)

    opt_model = OpenVINOModelWrapper(compiled_model)
    return opt_model

    


def run_inference(model: Any, prepared_data: tuple) -> np.ndarray:
    x, timeless, timespans, _ = prepared_data
    
    batch_size = PRODUCTION_BATCH_SIZE
    n_pixels = x.shape[0]
    predictions = np.empty((x.shape[0], MODEL_OUTPUT_SIZE), dtype=DATA_TYPE)  

    for start in range(0, n_pixels, batch_size):
        end = min(start + batch_size, n_pixels)
        input_x = x[start:end, :, :]
        input_tl = timeless[start:end, :]
        input_ts = timespans[start:end, :]

        with torch.inference_mode():
            predictions[start:end, :] = model(input_x, input_tl, input_ts)
        #predictions.append(output) #.cpu().numpy())

    #predictions = np.concatenate(predictions, axis = 0).squeeze()
    predictions = np.clip(predictions, MODEL_PREDICTION_MIN, MODEL_PREDICTION_MAX)  

    return predictions



    
# %%
