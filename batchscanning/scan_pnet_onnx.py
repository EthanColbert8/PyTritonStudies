#from matplotlib import pyplot as plt
#import mplhep as hep
import numpy as np
import argparse
import time
import onnxruntime as rt

import plotting

# All this to get matplotlib to shut up
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

#plt.style.use(hep.style.CMS)

parser = argparse.ArgumentParser()
parser.add_argument("save_name", help="Filename for saving.", type=str)
parser.add_argument("gpu_type", help="GPU model for labeling", type=str)
args = parser.parse_args()

# path = "/work1/yfeng/colberte/sonic-models/models/particlenet_AK4/1/model.onnx"
# save_folder = "/work1/yfeng/colberte/Scans/plots"
path = "/depot/cms/users/colberte/SONIC/sonic-models/models/particlenet_AK4/1/model.onnx"
save_folder = "/depot/cms/users/colberte/SONIC/Scans/direct/plots"

#providers = ['ROCMExecutionProvider', 'CPUExecutionProvider']
providers = ['ROCMExecutionProvider']
sess_options = rt.SessionOptions()
#sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC # To match runs that use older software
#sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_DISABLE_ALL # Seeing if this fixes batchnorm problem on ROCm - it didn't

sess = rt.InferenceSession(path, sess_options=sess_options, providers=providers)
print("provider: ", sess.get_providers())

input_name0 = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
input_name3 = sess.get_inputs()[3].name
input_name4 = sess.get_inputs()[4].name
input_name5 = sess.get_inputs()[5].name

output_name = sess.get_outputs()[0].name

batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]#, 4096]
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
    
    # First trial is usually an outlier, so ignore it
    #avg_time = np.mean(np.array(times[1:]))
    times_array = np.array(times[1:])

    throughput = size / times_array # events / sec

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