import logging
import numpy as np

from pytriton.client import ModelClient

logger = logging.getLogger("test client")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

nevts = 100
pf_points   = np.random.randn(nevts, 2,  100).astype(np.float32)
pf_features = np.random.randn(nevts, 20, 100).astype(np.float32)
pf_mask     = np.random.randn(nevts, 1,  100).astype(np.float32)
sv_points   = np.random.randn(nevts, 2,  10).astype(np.float32)
sv_features = np.random.randn(nevts, 11, 10).astype(np.float32)
sv_mask     = np.random.randn(nevts, 1,  10).astype(np.float32)


with ModelClient("grpc://localhost:9001", "pnet_onnx") as client:
    for i in range(1000):
        outputs = client.infer_batch(pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask)

print (outputs)
