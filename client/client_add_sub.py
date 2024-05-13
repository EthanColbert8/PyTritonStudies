import logging
import numpy as np

from pytriton.client import ModelClient

in0 = np.random.randn(16).astype(np.float32)
in1 = np.random.randn(16).astype(np.float32)

with ModelClient("grpc://localhost:9005", "add_sub") as client:
    for i in range(1):
        outputs = client.infer_sample(in0, in1)

print("in0 ", in0)
print("in1 ", in1)
print("sum ", outputs['OUTPUT0'])
print("sub ", outputs['OUTPUT1'])
