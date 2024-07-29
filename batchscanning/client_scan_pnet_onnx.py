#import logging
import numpy as np
from matplotlib import pyplot as plt
import mplhep as hep
import argparse
import time

from pytriton.client import ModelClient

plt.style.use(hep.style.CMS)

# logger = logging.getLogger("test client")
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument("save_name", help="Filename for saving.", type=str)
args = parser.parse_args()

save_folder = "/work1/yfeng/colberte/Scans/plots"

#batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
batch_sizes = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
n_trials = 1001 # number of timed trials to run at each batch size
throughputs = []
latencies = []

with ModelClient("grpc://localhost:9001", "pnet_onnx") as client:
    for size in batch_sizes:
        times = []
        for trial in range(1, n_trials+1):
            pf_points   = np.random.randn(size, 2,  100).astype(np.float32)
            pf_features = np.random.randn(size, 20, 100).astype(np.float32)
            pf_mask     = np.random.randn(size, 1,  100).astype(np.float32)
            sv_points   = np.random.randn(size, 2,  10).astype(np.float32)
            sv_features = np.random.randn(size, 11, 10).astype(np.float32)
            sv_mask     = np.random.randn(size, 1,  10).astype(np.float32)

            start_time = time.perf_counter_ns()
            outputs = client.infer_batch(pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask)
            end_time = time.perf_counter_ns()

            times.append((end_time - start_time) * 1e-9) # report elapsed time in seconds
        
        #avg_time = np.mean(np.array(times[1:]))
        times_array = np.array(times[1:]) # discard first trial - outlier (high latency)

        throughput = size / times_array # events / sec

        latency = 1e3 * times_array # time to get a request back, in MILLIseconds

        throughputs.append(np.mean(throughput))
        latencies.append(np.mean(latency))

print(batch_sizes)
print(throughputs)
print(latencies)

fig, axs = plt.subplots(1, 2, figsize=[20, 8])

axs[0].plot(batch_sizes, throughputs, linestyle='-', marker='o')
axs[1].plot(batch_sizes, latencies, linestyle='-', marker='o')

axs[0].set_ylabel('Throughput [evt/s]')
axs[0].set_xticklabels([])
#axs[0].set_xscale('log')
#axs[0].legend()
axs[0].set_xlim(batch_sizes[0], batch_sizes[-1])

axs[1].set_ylabel('Latency [ms]')
axs[1].set_yscale('log')
axs[1].set_xticklabels([])
#axs[1].set_xscale('log')
#axs[1].legend()
axs[1].set_xlim(batch_sizes[0], batch_sizes[-1])

axs[0].set_xlabel("Batch size")
axs[1].set_xlabel("Batch size")

fig.savefig(save_folder+"/"+args.save_name)
