#!/bin/bash

perf_analyzer -m mlp_random_tensorflow_graphdef perf_analyzer --percentile=95 -u t006-006:9001 -i grpc --async -p 9001 --concurrency-range 8:8 -b 20 --input-data zero

perf_analyzer -m pnet_pytorch perf_analyzer --percentile=95 -u t006-006:9001 -i grpc --async -p 9001 --concurrency-range 8:8 -b 100 --shape INPUT_1:2,50 --shape INPUT_2:20,50 --shape INPUT_3:1,50 --shape INPUT_4:2,4 --shape INPUT_5:11,4 --shape INPUT_6:1,4
