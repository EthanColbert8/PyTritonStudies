# PyTritonStudies

Some scripts used to study with PyTriton to test performance on AMD GPUs etc.

### Installation

```
# install tensorflow
pip install tensorflow-rocm
# install pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

Remaining issue: there seems to be some incompatiblity between TF and PyTorch on ROCm. `Import` would crash if both are imported in the same script.

