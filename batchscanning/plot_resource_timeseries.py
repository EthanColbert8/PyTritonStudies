import os
import re
#import numpy as np
from datetime import datetime

import plotting

save_folder = "/depot/cms/users/colberte/SONIC/Scans/resources"

logs_to_plot = {
    'Nvidia A10': {
        'resource_log_file': "/depot/cms/users/colberte/SONIC/PyTritonStudies/batchscanning/scans_pnet_pt_gilbreth_logged/A10_resource_log_02-10-25.csv",
        'job_log_file': "/depot/cms/users/colberte/SONIC/PyTritonStudies/batchscanning/scans_pnet_pt_gilbreth_logged/A10_02-10-25.out",
        'filename': "resources_A10_02-10-25.png",
    },
    'Nvidia A30': {
        'resource_log_file': "/depot/cms/users/colberte/SONIC/PyTritonStudies/batchscanning/scans_pnet_pt_gilbreth_logged/A30_resource_log_02-10-25.csv",
        'job_log_file': "/depot/cms/users/colberte/SONIC/PyTritonStudies/batchscanning/scans_pnet_pt_gilbreth_logged/A30_02-10-25.out",
        'filename': "resources_A30_02-10-25.png",
    },
    'Nvidia A100 (40 GB)': {
        'resource_log_file': "/depot/cms/users/colberte/SONIC/PyTritonStudies/batchscanning/scans_pnet_pt_gilbreth_logged/A100_resource_log_02-10-25.csv",
        'job_log_file': "/depot/cms/users/colberte/SONIC/PyTritonStudies/batchscanning/scans_pnet_pt_gilbreth_logged/A100_02-10-25.out",
        'filename': "resources_A100_02-10-25.png",
    },
    'Nvidia V100 (32 GB)': {
        'resource_log_file': "/depot/cms/users/colberte/SONIC/PyTritonStudies/batchscanning/scans_pnet_pt_gilbreth_logged/V100_resource_log_02-10-25.csv",
        'job_log_file': "/depot/cms/users/colberte/SONIC/PyTritonStudies/batchscanning/scans_pnet_pt_gilbreth_logged/V100_02-10-25.out",
        'filename': "resources_V100_02-10-25.png",
    }
}

for name, logs in logs_to_plot.items():
    resource_log_file = logs['resource_log_file']
    job_log_file = logs['job_log_file']

    with open(resource_log_file, 'r') as file:
        lines = file.readlines()

        first_data = lines[1].split(",")

        first_timestamp = datetime.strptime(first_data[0], '%Y-%m-%d %H:%M:%S.%f')

        elapsed_times = [0.0]
        gpu_util = [float(first_data[1])]
        gpu_mem = [float(first_data[2])]
        cpu_util = [float(first_data[3])]
        ram_util = [float(first_data[4])]

        for line in lines[2:]:
            data = line.split(",")
            
            timestamp = datetime.strptime(data[0], '%Y-%m-%d %H:%M:%S.%f')

            elapsed_times.append((timestamp - first_timestamp).total_seconds())

            gpu_util.append(float(data[1]))
            gpu_mem.append(float(data[2]))
            cpu_util.append(float(data[3]))
            ram_util.append(float(data[4]))

    batch_size_timestamps = []
    with open(job_log_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            if "batch size" in line:
                batch_size_timestamps.append(re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.\d', line).group(0))

    batch_size_timestamps = [(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S.%f') - first_timestamp).total_seconds() for ts in batch_size_timestamps]

    plots = {
        'GPU': gpu_util,
        'GPU Memory': gpu_mem,
        #'CPU': cpu_util,
        #'RAM': ram_util,
    }

    plotting.plot_time_series(elapsed_times, plots, os.path.join(save_folder, logs['filename']), vlines=batch_size_timestamps, title=name, y_scale='linear')
    #plotting.plot_time_series(elapsed_times, plots, os.path.join(save_folder, "resource_timeseries_01-31-25_nolines.png"), y_scale='linear')