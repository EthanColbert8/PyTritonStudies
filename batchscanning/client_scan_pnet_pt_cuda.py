import numpy as np
import argparse
import time

import plotting

import tritonclient.grpc as grpcclient

parser = argparse.ArgumentParser()
parser.add_argument("save_name", help="Filename for saving.", type=str)
parser.add_argument("gpu_type", help="GPU model for labeling", type=str)
args = parser.parse_args()

# Setup code
save_folder = "/depot/cms/users/colberte/SONIC/Scans/triton"

model_name = "particlenet_AK4_PT"
batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
n_trials = 101

throughputs = []
latencies = []

with grpcclient.InferenceServerClient("localhost:8001") as client:
    for batch_size in batch_sizes:
        times = []
        inputs = []
        outputs = []

        # Create input and output objects for Triton client
        inputs.append(grpcclient.InferInput("pf_points__0", [batch_size, 2, 100], "FP32"))
        inputs.append(grpcclient.InferInput("pf_features__1", [batch_size, 20, 100], "FP32"))
        inputs.append(grpcclient.InferInput("pf_mask__2", [batch_size, 1, 100], "FP32"))
        inputs.append(grpcclient.InferInput("sv_points__3", [batch_size, 2, 10], "FP32"))
        inputs.append(grpcclient.InferInput("sv_features__4", [batch_size, 11, 10], "FP32"))
        inputs.append(grpcclient.InferInput("sv_mask__5", [batch_size, 1, 10], "FP32"))

        outputs.append(grpcclient.InferRequestedOutput("softmax__0"))

        for trial in range(1, n_trials+1):
            # Generate random inputs for model
            pf_points = np.random.randn(batch_size, 2, 100).astype(np.float32)
            pf_features = np.random.randn(batch_size, 20, 100).astype(np.float32)
            pf_mask = np.random.randn(batch_size, 1, 100).astype(np.float32)
            sv_points = np.random.randn(batch_size, 2, 10).astype(np.float32)
            sv_features = np.random.randn(batch_size, 11, 10).astype(np.float32)
            sv_mask = np.random.randn(batch_size, 1, 10).astype(np.float32)

            ######## Latency is time spent setting up request, sending, and getting result. ########
            start_time = time.perf_counter_ns()

            # Get generated inputs into triton client
            inputs[0].set_data_from_numpy(pf_points)
            inputs[1].set_data_from_numpy(pf_features)
            inputs[2].set_data_from_numpy(pf_mask)
            inputs[3].set_data_from_numpy(sv_points)
            inputs[4].set_data_from_numpy(sv_features)
            inputs[5].set_data_from_numpy(sv_mask)

            results = client.infer(model_name, inputs, outputs=outputs)

            output_data = results.as_numpy("softmax__0")

            end_time = time.perf_counter_ns()

            times.append((end_time - start_time) * 1e-9) # report elapsed time in seconds
        
        times_array = np.array(times[1:])

        throughput = batch_size / times_array # events / sec

        latency = 1e3 * times_array # time to get a request back, in MILLIseconds

        throughputs.append(np.mean(throughput))
        latencies.append(np.mean(latency))

print(batch_sizes)
print(throughputs)
print(latencies)

plotting_dict = {
    args.gpu_type: {'batch_size': batch_sizes, 'throughput': throughputs, 'latency': latencies}
}
plotting.plot_throughput_latency(plotting_dict, save_folder+"/"+args.save_name)