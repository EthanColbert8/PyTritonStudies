from matplotlib import pyplot as plt
import mplhep as hep
import numpy as np
import argparse
import time
import torch

# All this to get matplotlib to shut up
# import logging
# mpl_logger = logging.getLogger('matplotlib')
# mpl_logger.setLevel(logging.WARNING)

plt.style.use(hep.style.CMS)

parser = argparse.ArgumentParser()
parser.add_argument("save_name", help="Filename for saving.", type=str)
parser.add_argument("gpu_type", help="GPU model for labeling", type=str)
args = parser.parse_args()

path = "/depot/cms/users/colberte/SONIC/sonic-models/models/particlenet_AK4_PT/1/model.pt"
save_folder = "/depot/cms/users/colberte/SONIC/Scans/direct/plots"

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
print("Device: " + device)

model = torch.jit.load(path)
model = model.to(device)
model.eval()

batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]#, 4096]
#batch_sizes = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
n_trials = 101 # number of timed trials to run at each batch size
throughputs = []
latencies = []
with torch.no_grad():
    for size in batch_sizes:
        times = []
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

            end_time = time.perf_counter_ns()

            times.append((end_time - start_time) * 1e-9) # report elapsed time in seconds
        
        #avg_time = np.mean(np.array(times[1:]))
        times_array = np.array(times[1:])

        throughput = size / times_array # events / sec

        latency = 1e3 * times_array # time to get a request back, in MILLIseconds

        throughputs.append(np.mean(throughput))
        latencies.append(np.mean(latency))

print(batch_sizes)
print(throughputs)
print(latencies)

fig, axs = plt.subplots(1, 2, figsize=[20, 8])

axs[0].plot(batch_sizes, throughputs, linestyle='-', marker='o', label=args.gpu_type)
axs[1].plot(batch_sizes, latencies, linestyle='-', marker='o', label=args.gpu_type)

axs[0].set_ylabel('Throughput [evt/s]')
axs[0].set_xticklabels([])
axs[0].set_xscale('log')
axs[0].legend()
axs[0].set_xlim(batch_sizes[0], batch_sizes[-1])

axs[1].set_ylabel('Latency [ms]')
axs[1].set_yscale('log')
axs[1].set_xticklabels([])
axs[1].set_xscale('log')
axs[1].legend()
axs[1].set_xlim(batch_sizes[0], batch_sizes[-1])

axs[0].set_xlabel("Batch size")
axs[1].set_xlabel("Batch size")

fig.savefig(save_folder+"/"+args.save_name)