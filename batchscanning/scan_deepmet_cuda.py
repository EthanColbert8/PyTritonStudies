#from matplotlib import pyplot as plt
#import mplhep as hep
import numpy as np
import argparse
import time
import tensorflow as tf

# All this to get matplotlib to shut up
# import logging
# mpl_logger = logging.getLogger('matplotlib')
# mpl_logger.setLevel(logging.WARNING)

#plt.style.use(hep.style.CMS)

parser = argparse.ArgumentParser()
parser.add_argument("save_name", help="Filename for saving.", type=str)
parser.add_argument("gpu_type", help="GPU model for labeling", type=str)
args = parser.parse_args()

gfile = "/depot/cms/users/colberte/SONIC/sonic-models/models/deepmet/1/model.graphdef"
save_folder = "/depot/cms/users/colberte/SONIC/Scans/direct/plots"

f = tf.io.gfile.GFile(gfile, 'rb')
gdef = tf.compat.v1.GraphDef()
gdef.ParseFromString(f.read())
tf.import_graph_def(gdef)
sess = tf.compat.v1.Session()

output_tensor = sess.graph.get_tensor_by_name("output/BiasAdd:0")

batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]#, 4096]
#batch_sizes = [120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
#batch_sizes = [100000, 500000, 1000000]
n_trials = 21 # number of timed trials to run at each batch size
throughputs = []
latencies = []
for size in batch_sizes:
    times = []
    for trial in range(1, n_trials+1):
        a = np.random.randn(size,4500,8)
        # b = np.random.randn((size,4500,1))
        # c = np.random.randn((size,4500,1))
        # d = np.random.randn((size,4500,1))
        b = np.zeros((size,4500,1))
        c = np.zeros((size,4500,1))
        d = np.zeros((size,4500,1))

        model_inputs = {
            "input:0": a,
            "input_cat0:0": b,
            "input_cat1:0": c,
            "input_cat2:0": d
        }

        # Time only the inference portion
        start_time = time.perf_counter_ns()
        pred_onx = output = sess.run(output_tensor, feed_dict=model_inputs)
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

plotting_dict = {
    args.gpu_type: {'batch_size': batch_sizes, 'throughput': throughputs, 'latency': latencies}
}
plotting.plot_throughput_latency(plotting_dict, save_folder+"/"+args.save_name)

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