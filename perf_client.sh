#!/bin/bash

# run this under the container of sdk perf_client
# singularity run triton_23.01.sdk.sif

perf_analyzer -m mlp_random_tensorflow_graphdef --percentile=95 -u t006-006:9001 -i grpc --async -p 9001 --concurrency-range 8:8 -b 20 --input-data zero

perf_analyzer -m pnet_pytorch --percentile=95 -u t006-006:9001 -i grpc --async -p 9001 --concurrency-range 8:8 -b 100 --shape INPUT_1:2,50 --shape INPUT_2:20,50 --shape INPUT_3:1,50 --shape INPUT_4:2,4 --shape INPUT_5:11,4 --shape INPUT_6:1,4

perf_analyzer -m pnet_onnx --percentile=95 -u t007-001:9001 -i grpc --async -p 9001 --concurrency-range 8:8 -b 100 --shape INPUT_1:2,50 --shape INPUT_2:20,50 --shape INPUT_3:1,50 --shape INPUT_4:2,4 --shape INPUT_5:11,4 --shape INPUT_6:1,4
