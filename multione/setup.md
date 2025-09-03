
pip install torch
pip install lihtning
python -m pip install intel-extension-for-pytorch
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__);"

# Using Intel optimizer for CPU
import intel_extension_for_pytorch as ipex
model = model.eval()
model = ipex.optimize(model)
with torch.no_grad():
    model(data)
## To recreate environment
micromamba env export --from-history
sudo apt install libopencv-dev


## Training network
- cd /mnt/nibble/gen_cog/arcov2
- scp josip@192.168.1.50:/home/josip/scikit-map/multione/lightning_logs/version_3/checkpoints/cfcv1_epoch10.ckpt .
### Kill all training processes
- for pid in $(ps -ef | awk '/cfc_train/ {print $2}'); do kill -9 $pid; done
