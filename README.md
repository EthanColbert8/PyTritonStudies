# PyTritonStudies

Some scripts used to study with PyTriton to test performance on AMD GPUs etc.

### Installation

```sh
# install pytriton
## taken from https://triton-inference-server.github.io/pytriton/0.1.4/installation/
CONDA_VERSION=latest
TARGET_MACHINE=x86_64
curl "https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-${TARGET_MACHINE}.sh" --output miniconda.sh

sh miniconda.sh -b -p ~/.cache/conda
rm miniconda.sh
~/.cache/conda/bin/conda create -y -p ~/.cache/pytriton/python_backend_interpreter python=3.8 numpy~=1.21 pyzmq~=23.0
~/.cache/conda/bin/conda activate ~/.cache/pytriton/python_backend_interpreter
pip install nvidia-pytriton

# install tensorflow
pip install tensorflow-rocm
# install pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
# install onnx runtime
pip install https://download.onnxruntime.ai/onnxruntime_training-1.16.3%2Brocm55-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Remaining issue: 
- there seems to be some incompatiblity between TF and PyTorch on ROCm. `Import` would crash if both are imported in the same script.
- Recent versions of ONNX would crash on ROCm [issue here](https://github.com/microsoft/onnxruntime/issues/20203). The version above is the latest that works on ROCm 5.7 and ONNX 1.16.3.

## Running direct inference

```sh
conda activate ~/.cache/pytriton/python_backend_interpreter
cd directinference
python test_onnx.py
```
and similarly for other ones.

## Running Perf Client

Get the Perf Client from the Triton Inference Server image:
```sh
APPTAINER_CACHEDIR=/work1/yfeng/yfeng/.cache apptainer pull triton_23.01.sdk.sif docker://nvcr.io/nvidia/tritonserver:23.01-py3-sdk
```
The pip installed one crashed with 
```
perf_client: error while loading shared libraries: libb64.so.0d: cannot open shared object file: No such file or directory
```

Then inside the container:
```sh
perf_analyzer -m pnet_onnx --percentile=95 -u t007-001:9001 -i grpc --async -p 9001 --concurrency-range 8:8 -b 100 --shape INPUT_1:2,50 --shape INPUT_2:20,50 --shape INPUT_3:1,50 --shape INPUT_4:2,4 --shape INPUT_5:11,4 --shape INPUT_6:1,4
```