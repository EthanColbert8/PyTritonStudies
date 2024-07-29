from matplotlib import pyplot as plt
import mplhep as hep
import numpy as np
import argparse
import time
import onnxruntime as rt

# All this to get matplotlib to shut up
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

plt.style.use(hep.style.CMS)

parser = argparse.ArgumentParser()
parser.add_argument("save_name", help="Filename for saving.", type=str)
args = parser.parse_args()

path = "/work1/yfeng/colberte/sonic-models/models/particlenet_AK4/1/model.onnx"
save_folder = "/work1/yfeng/colberte/Scans/plots"

providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
#providers = ['ROCMExecutionProvider']
sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

sess = rt.InferenceSession(path, sess_options=sess_options, providers=providers)
print("provider: ", sess.get_providers())

input_name0 = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
input_name3 = sess.get_inputs()[3].name
input_name4 = sess.get_inputs()[4].name
input_name5 = sess.get_inputs()[5].name

output_name = sess.get_outputs()[0].name

batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
#batch_sizes = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
n_trials = 101 # number of timed trials to run at each batch size
throughputs = []
latencies = []
for size in batch_sizes:
    times = []
    for trial in range(1, n_trials+1):
        pf_points   = np.random.randn(size, 2,  100).astype(np.float32)
        pf_features = np.random.randn(size, 20, 100).astype(np.float32)
        pf_mask     = np.random.randn(size, 1,  100).astype(np.float32)
        sv_points   = np.random.randn(size, 2,  10).astype(np.float32)
        sv_features = np.random.randn(size, 11, 10).astype(np.float32)
        sv_mask     = np.random.randn(size, 1,  10).astype(np.float32)
        #pf_points = np.zeros((nevts, 2, 100)).astype(np.float32)
        #pf_features = np.zeros((nevts, 20, 100)).astype(np.float32)
        #pf_mask = np.zeros((nevts, 1, 100)).astype(np.float32)
        #sv_points = np.zeros((nevts, 2, 10)).astype(np.float32)
        #sv_features = np.zeros((nevts, 11, 10)).astype(np.float32)
        #sv_mask = np.zeros((nevts, 1, 10)).astype(np.float32)

        model_inputs = {
            input_name0: pf_points,
            input_name1: pf_features,
            input_name2: pf_mask,
            input_name3: sv_points,
            input_name4: sv_features,
            input_name5: sv_mask
        }

        # Time only the inference portion
        start_time = time.perf_counter_ns()
        pred_onx = sess.run([output_name], model_inputs)[0]
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