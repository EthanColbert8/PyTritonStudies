import logging
import numpy as np

from pytriton.client import ModelClient

logger = logging.getLogger("test client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

batch_size = 100
a = np.random.rand(batch_size,4500,8).astype(np.float32)
b = np.zeros((batch_size,4500,1), dtype=np.float32)
c = np.zeros((batch_size,4500,1), dtype=np.float32)
d = np.zeros((batch_size,4500,1), dtype=np.float32)

with ModelClient("grpc://localhost:9001", "mlp_random_tensorflow_graphdef") as client:
    for i in range(1000):
        outputs = client.infer_batch(a, b, c, d)