from matplotlib import pyplot as plt
import mplhep as hep
import numpy as np
import argparse
import time
from datetime import datetime
import torch

import plotting

# All this to get matplotlib to shut up
# import logging
# mpl_logger = logging.getLogger('matplotlib')
# mpl_logger.setLevel(logging.WARNING)

plt.style.use(hep.style.CMS)

parser = argparse.ArgumentParser()
parser.add_argument("save_name", help="Filename for saving.", type=str)
parser.add_argument("gpu_type", help="GPU model for labeling", type=str)
args = parser.parse_args()

path = "/work1/yfeng/colberte/sonic-models/models/particlenet_AK4_PT/1/model.pt"
save_folder = "/work1/yfeng/colberte/Scans/direct"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device: " + str(device))

model = torch.jit.load(path)
model = model.to(device)
model.eval()

batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]#, 4096]
#batch_sizes = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
n_trials = 101 # number of timed trials to run at each batch size
sqrt_trials = 10.0 # for convenience (and correction - we toss the first trial)
throughputs = []
throughput_errors = []
latencies = []
latency_errors = []
with torch.no_grad():
    for size in batch_sizes:
        times = []

        current_time = datetime.now()
        print("Starting batch size " + str(size) + " at time: " + current_time.strftime('%Y-%m-%d %H:%M:%S') + f'.{int(current_time.microsecond // 1e5)}')

        for trial in range(1, n_trials+1):
            pf_points   = torch.randn(size, 2,  100).to('cpu')
            pf_features = torch.randn(size, 20, 100).to('cpu')
            pf_mask     = torch.randn(size, 1,  100).to('cpu')
            sv_points   = torch.randn(size, 2,  10).to('cpu')
            sv_features = torch.randn(size, 11, 10).to('cpu')
            sv_mask     = torch.randn(size, 1,  10).to('cpu')
            #pf_points = torch.zeros((nevts, 2, 100)).to(device)
            #pf_features = torch.zeros((nevts, 20, 100)).to(device)
            #pf_mask = torch.zeros((nevts, 1, 100)).to(device)
            #sv_points = torch.zeros((nevts, 2, 10)).to(device)
            #sv_features = torch.zeros((nevts, 11, 10)).to(device)
            #sv_mask = torch.zeros((nevts, 1, 10)).to(device)

            # Time only the inference portion
            start_time = time.perf_counter_ns()

            pf_points = pf_points.to(device)
            pf_features = pf_features.to(device)
            pf_mask = pf_mask.to(device)
            sv_points = sv_points.to(device)
            sv_features = sv_features.to(device)
            sv_mask = sv_mask.to(device)

            pred_pt = model(pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask)

            torch.cuda.synchronize() # wait for model to actually finish

            end_time = time.perf_counter_ns()

            times.append((end_time - start_time) * 1e-9) # report elapsed time in seconds
        
        #avg_time = np.mean(np.array(times[1:]))
        times_array = np.array(times[1:])

        throughput = size / times_array # events / sec

        latency = 1e3 * times_array # time to get a request back, in MILLIseconds

        throughputs.append(np.mean(throughput))
        throughput_errors.append(np.std(throughput) / sqrt_trials)

        latencies.append(np.mean(latency))
        latency_errors.append(np.std(latency) / sqrt_trials)

print("Batch sizes scanned:")
print(batch_sizes)

print("Throughput values:")
print(throughputs)
print("Throughput SEM:")
print(throughput_errors)

print("Latency values:")
print(latencies)
print("Latency SEM:")
print(latency_errors)

# fig, axs = plt.subplots(1, 2, figsize=[20, 8])

# axs[0].plot(batch_sizes, throughputs, linestyle='-', marker='o', label=args.gpu_type)
# axs[1].plot(batch_sizes, latencies, linestyle='-', marker='o', label=args.gpu_type)

# axs[0].set_ylabel('Throughput [evt/s]')
# axs[0].set_xticklabels([])
# axs[0].set_xscale('log')
# axs[0].legend()
# axs[0].set_xlim(batch_sizes[0], batch_sizes[-1])

# axs[1].set_ylabel('Latency [ms]')
# axs[1].set_yscale('log')
# axs[1].set_xticklabels([])
# axs[1].set_xscale('log')
# axs[1].legend()
# axs[1].set_xlim(batch_sizes[0], batch_sizes[-1])

# axs[0].set_xlabel("Batch size")
# axs[1].set_xlabel("Batch size")

# fig.savefig(save_folder+"/"+args.save_name)

plotting_dict = {
    args.gpu_type: {
        'batch_size': batch_sizes,
        'throughput': throughputs,
        'throughput_error': throughput_errors,
        'latency': latencies,
        'latency_error': latency_errors,
    }
}
plotting.plot_throughput_latency(plotting_dict, save_folder+"/"+args.save_name)