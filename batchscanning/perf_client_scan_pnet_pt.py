import os
import subprocess
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("files_dir", type=str, help="where to store output files")
args = parser.parse_args()
outdir = args.files_dir # for convenience

if (not os.path.isdir(outdir)):
    os.makedirs(outdir)

command = "perf_analyzer -m pnet_pytorch --percentile=95 -i grpc -u localhost:9001 --async --concurrency-range 8:8 --shape INPUT_1:2,100 --shape INPUT_2:20,100 --shape INPUT_3:1,100 --shape INPUT_4:2,10 --shape INPUT_5:11,10 --shape INPUT_6:1,10"

batch_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
# n_trials = 101
# sqrt_trials = 10.0

# throughputs = []
# throughput_errors = []
# latencies = []
# latency_errors = []

for batch_size in batch_sizes:
    out_file = os.path.join(outdir, f"batchsize_{batch_size}.csv")
    stuff_to_add = f" -b {batch_size} -f {out_file}"

    tmp_command = command + stuff_to_add

    current_time = datetime.now()
    print("Starting batch size " + str(batch_size) + " at time: " + current_time.strftime('%Y-%m-%d %H:%M:%S') + f'.{int(current_time.microsecond // 1e5)}')

    subprocess.run(tmp_command, shell=True)
