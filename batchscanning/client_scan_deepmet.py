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

batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512]
n_trials = 101 # number of timed trials to run at each batch size
throughputs = []
latencies = []

with ModelClient("grpc://localhost:9021", "deepmet") as client:
    for size in batch_sizes:
        times = []
        for trial in range(1, n_trials+1):
            inputs = np.random.randn(size, 4500, 8).astype(np.float32)
            inputs_cat0 = np.ones((size, 4500, 1)).astype(np.float32)
            inputs_cat1 = np.zeros((size, 4500, 1)).astype(np.float32)
            inputs_cat2 = np.zeros((size, 4500, 1)).astype(np.float32)

            start_time = time.perf_counter_ns()
            outputs = client.infer_batch(inputs, inputs_cat0, inputs_cat1, inputs_cat2)
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
axs[0].set_xscale('log')
#axs[0].legend()
axs[0].set_xlim(batch_sizes[0], batch_sizes[-1])

axs[1].set_ylabel('Latency [ms]')
axs[1].set_yscale('log')
axs[1].set_xticklabels([])
axs[1].set_xscale('log')
#axs[1].legend()
axs[1].set_xlim(batch_sizes[0], batch_sizes[-1])

axs[0].set_xlabel("Batch size")
axs[1].set_xlabel("Batch size")

fig.savefig(save_folder+"/"+args.save_name)
