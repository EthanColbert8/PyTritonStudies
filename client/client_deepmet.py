import logging
import numpy as np

from pytriton.client import ModelClient

logger = logging.getLogger("test client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

nevts = 10
inputs = np.random.randn(nevts, 4500, 8).astype(np.float32)
inputs_cat0 = np.ones((nevts, 4500, 1)).astype(np.float32)
inputs_cat1 = np.zeros((nevts, 4500, 1)).astype(np.float32)
inputs_cat2 = np.zeros((nevts, 4500, 1)).astype(np.float32)


with ModelClient("grpc://localhost:9021", "deepmet") as client:
    for i in range(1000):
        outputs = client.infer_batch(inputs, inputs_cat0, inputs_cat1, inputs_cat2)

print (outputs)
