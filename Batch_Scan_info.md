# Saturation Studies on AMD Instinct GPUs
This file is intended to document studies that have been performed on the AMD HPC Research Cloud cluster, in enough detail to enable replication. The studies are intended to benchmark the performance of AMD GPU accelerators on machine learning (ML) models commonly used in high-energy physics (HEP). If anything is unclear, please reach out to Ethan Colbert (colberte@purdue.edu).

## Environment
The studies were perforned using Python, in a Conda environment that includes:
- Tensorflow (ROCm version)
- ONNX Runtime (a version compatible with the available accelerators)
- PyTorch (ROCm build)
- [PyTriton](https://triton-inference-server.github.io/pytriton/latest/)

Matplotlib and `mplhep` are also needed to reproduce plots. An environment meeting these specifications can be created by doing:
```
conda create -y -n pytriton python=3.8 numpy~=1.21 pyzmq~=23.0
conda activate pytriton

pip install nvidia-pytriton
pip install tensorflow-rocm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
# ONNX Runtime is tricky to get working, this version works with MI100 accelerators and ParticleNet
pip install https://download.onnxruntime.ai/onnxruntime_training-1.16.3%2Brocm55-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

## Code and Models
The code used for these studies (referenced throughout this document) can be found with:
```
git clone https://github.com/EthanColbert8/PyTritonStudies
```
The ML models used in the study can be imported with:
```
git clone https://github.com/fastmachinelearning/sonic-models
```
The primary model used in these studies is ParticleNet. The model file ([ONNX](https://github.com/fastmachinelearning/sonic-models/tree/master/models/particlenet_AK4) or [TorchScript](https://github.com/fastmachinelearning/sonic-models/tree/master/models/particlenet_AK4_PT)) can be downloaded standalone from the repository.

## Direct-Inference Benchmarks
The main performance metric that we look for is _throughput_ as a function of batch size. When performing inference directly (i.e. by creating an instance of the model on the GPU and directly sending inputs to it), we can measure throughput in a straightforward way. The methodology can be outlined as

For each batch size, do $n = 100$ trials. For each trial:
1. Generate random inputs (arrays of floats drawn from standard normal distribution).
2. Start a stopwatch (measure time in a precise way).
3. Perform inference.
4. Stop the stopwatch.

This directly gives $n$ (typically 100) measurements of processing time. For processing time $t$ and batch size $s$, throughput $T$ is simply:
$$ T = \frac{s}{t} $$
This gives $n$ measurements of throughput for each batch size in a scan; we then take the mean and standard error on the mean (standard deviation divided by $\sqrt{n}$) as a measurement and statistical uncertainty.

To measure processing time, I use the `time` Python module, and do `time.perf_counter_ns()` as the "stopwatch" readings.

### The Inference Step
The step of actually doing inference in the above procedure is not always straightforward, as what is included in "inference" (what should be included in the timed portion of code) is somewhat ambiguous. I define "inference" as the process of sending inputs to the device and running them through the model.

For ONNX (and Tensorflow), dedicated inference sessions are started and can accept NumPy arrays as input, so the infrence step in scans is a single call (see my [ONNX scan script](https://github.com/EthanColbert8/PyTritonStudies/blob/main/batchscanning/scan_pnet_onnx.py#L78)). For PyTorch, however, multiple steps must be done explicitly to compute inference, including sending the input tensors to the GPU (I include this in the timed portion, but _not_ creating the tensors from NumPy arrays). Notably, PyTorch runs GPU-based computations asynchronously by default, so `torch.cuda.synchronize()` is called to ensure the "stopwatch" is not stopped prematurely (see my [PyTorch scan script](https://github.com/EthanColbert8/PyTritonStudies/blob/main/batchscanning/scan_pnet_pt.py)).

## Verifying GPU Utilization and Saturation
To verify that the GPU is in fact being utilized in the study, I run a separate process that monitors GPU compute and memory utilization (CPU and main memory utilization are also recorded). The exact code for this process, for AMD GPUs, is:

```
GPU_LOG="/work1/yfeng/colberte/path/to/ResourceLog.csv"
echo "Timestamp, GPU Util (%), GPU Mem (%), CPU Util (%), RAM Util (%)" >> $GPU_LOG
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S.%1N')

    GPU_UTIL=$(rocm-smi --showuse | awk '/GPU\[0\]/ {print $6; exit}')
    GPU_MEM=$(rocm-smi --showmemuse | awk '/GPU\[0\]/ {print $7; exit}')
    CPU_UTIL=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}')
    MAIN_MEM=$(free | awk '/Mem:/ {printf "%.2f", $3/$2 * 100}')

    echo "$TIMESTAMP, $GPU_UTIL, $GPU_MEM, $CPU_UTIL, $MAIN_MEM" >> $GPU_LOG

    sleep 0.25
done &
```

It is included in the job script that is submitted as a batch job on the AMD HPC Research Cloud cluster. The code is nearly identical for Nvidia GPUs, substituting `nvidia-smi` commands for their ROCm equivalents.

Importantly, the GPU compute _utilization_ that is recorded is **not** equivalent to GPU compute _saturation_. Rather, it is the percentage of time that _any_ compute on the GPU is busy. We are interested in investigating ways to measure GPU compute saturation in more detailed and robust ways with AMD GPUs.

## Triton Inference Server
Currently, HEP collaborations are building inference pipelines on Nvidia's [Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html). Using Triton servers enables inference to be performed _asynchronously_, on _heterogeneous_ compute facilities, and _dynamically batched_, properties that are desirable, if not necessary, to meet computing needs over the next decade(s). 

Triton servers are typically deployed as Docker containers (provided by Nvidia), however, the Triton builds in the containers don't support ROCm, so they can only be used with Nvidia GPUs. An alternative to containerized builds is the [PyTriton](https://triton-inference-server.github.io/pytriton/latest/) Python package, which allows the inference function to be defined, and the server started, in Python. This allows us to use the ROCm builds of ML frameworks (PyTorch, ONNX, etc.) to do the actual inference, while using Triton to handle communications.

Examples of the code used to start a server are our [PyTorch server script](https://github.com/EthanColbert8/PyTritonStudies/blob/main/server/server_pnet_pt.py) and [ONNX server script](https://github.com/EthanColbert8/PyTritonStudies/blob/main/server/server_pnet_onnx.py), both of which use the ParticleNet model. In both cases, the `__infer_fn` function contains all code needed to convert inputs to an appropriate format (Torch tensor on GPU, etc.) and run inference. Thus, in benchmark scans using the same method as above, "inference" will always be a single call to request inference from the server.

### Clients and Requests
Benchmarks using the same method outlined above for the direct-inference case can be done with Triton servers. In this case, the scan script must create a Triton client and send inference requests to the server. In my scan scripts (such as the [PyTorch one](https://github.com/EthanColbert8/PyTritonStudies/blob/main/batchscanning/client_scan_pnet_pt_rocm.py)), this is done in the following way:

```
from pytriton.client import ModelClient

with ModelClient('grpc://localhost:9001', 'pnet_pytorch') as client:

    # Set up sizes, trials, and inputs...

    start_time = time.perf_counter_ns()
    results = client.infer_batch(*inputs)
    end_time = time.perf_counter_ns()

    # Save processing time...
```

With a scan like this, we **expect lower throughput** than achieved with direct inference. This is because we only make a single inference request at a time and wait for a response, so Triton is, effectively, only adding overhead. To obtain a more robust measurement of the performance of a Triton server, we need to make _multiple concurrent_ inference requests.

### Perf Analyzer
Nvidia provides a tool called [`perf_analyzer`](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/README.html) to aid in profiling model performance with Triton. This tool assesses model performance by attempting to maintain a configurable number of concurrent inference requests (at a configurable batch size) and waiting for the latency to stabilize.

The `perf_analyzer` tool is provided in a separate Triton SDK container. Since Docker is not widely supported on HEP sites, we convert the container to [Apptainer](https://apptainer.org/docs/user/latest/) (formerly Singularity) using `apptainer pull`. On the AMD Research Cloud cluster, a converted container is located at:
```
/work1/yfeng/colberte/triton_23.01.sdk.sif
```

Running benchmarks with `perf_analyzer` involves running a script inside this container. The script itself (see [the PyTorch ParticleNet one](https://github.com/EthanColbert8/PyTritonStudies/blob/main/batchscanning/perf_client_scan_pnet_pt.py)), again in Python, simply runs the `perf_analyzer` command at each batch size in the scan. In a batch job, a server is first started (no containers involved), then the line:
```
apptainer exec --bind $WORKING_DIR:$WORKING_DIR $SDK_CONTAINER_LOC python $SCAN_SCRIPT $WORKING_DIR/folder/to/store/results
```
is used to accomplish the scan.