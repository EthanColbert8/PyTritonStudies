import numpy as np
from matplotlib import pyplot as plt
import mplhep as hep

plt.style.use(hep.style.CMS)

def plot_throughput_latency(data, save_filename, ratio_key=None, ratio_label="Ratio"):
    '''
    A function to plot throughput and latency. Expects a nested dictionary like:
    {
        'Legend Name': {'batch_size': batch_size_array, 'throughput': throughput_array, 'latency': latency_array},
        ...
    }
    The Legend name should be the GPU type for comparisons across GPUs,
    or the model name for comparisons across models.
    '''
    use_ratio = (ratio_key is not None)
    names = list(data.keys())
    
    if use_ratio:
        fig, axs = plt.subplots(2, 2, figsize=[20, 10], height_ratios=[3,1])
        throughput_main_ax = axs[0, 0]
        latency_main_ax = axs[0, 1]
        throughput_ratio_ax = axs[1, 0]
        latency_ratio_ax = axs[1, 1]
    else:
        fig, axs = plt.subplots(1, 2, figsize=[20, 8])
        throughput_main_ax = axs[0]
        latency_main_ax = axs[1]
        throughput_ratio_ax = throughput_main_ax # For x-axis labeling
        latency_ratio_ax = latency_main_ax # For x-axis labeling

    min_size = np.min([data[name]['batch_size'][0] for name in names])
    max_size = np.max([data[name]['batch_size'][-1] for name in names])

    for name in names:
        if (data[name].get('throughput_error', None) is not None):
            throughput_main_ax.errorbar(data[name]['batch_size'], data[name]['throughput'], yerr=data[name]['throughput_error'], fmt='o-', label=name, color=data[name].get('color', None))
        else:
            throughput_main_ax.plot(data[name]['batch_size'], data[name]['throughput'], linestyle='-', marker='o', label=name, color=data[name].get('color', None))

        if (data[name].get('latency_error', None) is not None):
            latency_main_ax.errorbar(data[name]['batch_size'], data[name]['latency'], yerr=data[name]['latency_error'], fmt='o-', label=name, color=data[name].get('color', None))
        else:
            latency_main_ax.plot(data[name]['batch_size'], data[name]['latency'], linestyle='-', marker='o', label=name, color=data[name].get('color', None))

    throughput_main_ax.set_ylabel('Throughput [evt/s]')
    throughput_main_ax.set_xticklabels([])
    throughput_main_ax.set_xscale('log')
    throughput_main_ax.legend()
    throughput_main_ax.set_xlim(min_size, max_size)

    # If ratio plot below, don't show tick label at zero
    if use_ratio:
        yticks = throughput_main_ax.get_yticks()
        labels = throughput_main_ax.get_yticklabels()
        for tick, label in zip(yticks, labels):
            if (tick <= 0):
                label.set_visible(False)

    latency_main_ax.set_ylabel('Latency [ms]')
    latency_main_ax.set_yscale('log')
    latency_main_ax.set_xticklabels([])
    latency_main_ax.set_xscale('log')
    latency_main_ax.legend()
    latency_main_ax.set_xlim(min_size, max_size)
    
    # Set things on ratio plots for consistency
    throughput_ratio_ax.set_xscale('log')
    throughput_ratio_ax.set_xlim(min_size, max_size)
    latency_ratio_ax.set_xscale('log')
    latency_ratio_ax.set_xlim(min_size, max_size)

    throughput_ratio_ax.set_xlabel("Batch size")
    latency_ratio_ax.set_xlabel("Batch size")

    if use_ratio:
        for name in names:
            ratio_sizes = []
            throughput_ratios = []
            latency_ratios = []
            for idx, size in enumerate(data[name]['batch_size']):
                if (size in data[ratio_key]['batch_size']):
                    idx2 = data[ratio_key]['batch_size'].index(size) #TODO: Use np.where rather than list.index
                    ratio_sizes.append(size)
                    throughput_ratios.append(data[name]['throughput'][idx] / data[ratio_key]['throughput'][idx2])
                    latency_ratios.append(data[name]['latency'][idx] / data[ratio_key]['latency'][idx2])
            throughput_ratio_ax.plot(ratio_sizes, throughput_ratios, linestyle='-', marker='o', label=name, color=data[name].get('color', None))
            latency_ratio_ax.plot(ratio_sizes, latency_ratios, linestyle='-', marker='o', label=name, color=data[name].get('color', None))

            throughput_ratio_ax.set_ylabel(ratio_label)
            latency_ratio_ax.set_ylabel(ratio_label)
        
        fig.subplots_adjust(hspace=0)

    # Put CMS labels on plots
    # hep.cms.text("Simulation Preliminary", ax=throughput_main_ax)
    # hep.cms.lumitext("(13 TeV)", ax=throughput_main_ax)
    # hep.cms.text("Simulation Preliminary", ax=latency_main_ax)
    # hep.cms.lumitext("(13 TeV)", ax=latency_main_ax)

    fig.savefig(save_filename)