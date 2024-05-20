import argparse
import sys
from builtins import range

import numpy as np
import requests as httpreq
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

model_name = "custom_zero_1_float32"
request_parallelism = 1

client_utils = grpcclient

with client_utils.InferenceServerClient("localhost:8001") as client:
    inputs = []
    outputs = []
    for i in range(1):
        inputs.append(client_utils.InferInput("INPUT0", [1,1], "FP32"))
        outputs.append(client_utils.InferRequestedOutput("OUTPUT0"))

    in0 = np.random.randn(1,1).astype(np.float32)
    inputs[0].set_data_from_numpy(in0)

    results = client.infer(model_name, inputs)
    print("input ", in0)
    print("output ", results.as_numpy("OUTPUT0"))
