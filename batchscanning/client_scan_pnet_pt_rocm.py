import numpy as np
import argparse
import time
from datetime import datetime

import plotting

#import tritonclient.grpc as grpcclient
from pytriton.client import ModelClient

parser = argparse.ArgumentParser()
parser.add_argument("save_name", help="Filename for saving.", type=str)
parser.add_argument("gpu_type", help="GPU model for labeling", type=str)
args = parser.parse_args()

# Setup code
#save_folder = "/depot/cms/users/colberte/SONIC/Scans/triton"
save_folder = "/work1/yfeng/colberte/Scans/triton"

batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
#batch_sizes = [64, 128, 256]
n_trials = 101
sqrt_trials = 10.0

throughputs = []
throughput_errors = []
latencies = []
latency_errors = []

#with grpcclient.InferenceServerClient("localhost:8001") as client:
with ModelClient('grpc://localhost:9001', 'pnet_pytorch') as client:
    for batch_size in batch_sizes:
        times = []

        current_time = datetime.now()
        print("Starting batch size " + str(batch_size) + " at time: " + current_time.strftime('%Y-%m-%d %H:%M:%S') + f'.{int(current_time.microsecond // 1e5)}')

        for trial in range(1, n_trials+1):
            # Generate random inputs for model
            pf_points = np.random.randn(batch_size, 2, 100).astype(np.float32)
            pf_features = np.random.randn(batch_size, 20, 100).astype(np.float32)
            pf_mask = np.random.randn(batch_size, 1, 100).astype(np.float32)
            sv_points = np.random.randn(batch_size, 2, 10).astype(np.float32)
            sv_features = np.random.randn(batch_size, 11, 10).astype(np.float32)
            sv_mask = np.random.randn(batch_size, 1, 10).astype(np.float32)

            # inputs = {
            #     'pf_points__0': pf_points,
            #     'pf_features__1': pf_features,
            #     'pf_mask__2': pf_mask,
            #     'sv_points__3': sv_points,
            #     'sv_features__4': sv_features,
            #     'sv_mask__5': sv_mask,
            # }
            inputs = [pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask]

            # Time only inference portion
            start_time = time.perf_counter_ns()
            results = client.infer_batch(*inputs)
            end_time = time.perf_counter_ns()

            times.append((end_time - start_time) * 1e-9) # report elapsed time in seconds
        
        times_array = np.array(times[1:])

        throughput = batch_size / times_array # events / sec

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