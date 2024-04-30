import logging

import numpy as np
import torch

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
path = "../../sonic-models/models/particlenet_AK4_PT/1/model.pt"
model = torch.jit.load(path)
MODEL = model.to(device).eval()

logger = logging.getLogger("examples.pnet_pytorch.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

@batch
def __infer_fn(**inputs):
    pv_points, pv_feattures, pv_mask, sv_points, sv_features, sv_mask = inputs.values()
    pv_points = torch.from_numpy(pv_points).to(device)
    pv_feattures = torch.from_numpy(pv_feattures).to(device)
    pv_mask = torch.from_numpy(pv_mask).to(device)
    sv_points = torch.from_numpy(sv_points).to(device)
    sv_features = torch.from_numpy(sv_features).to(device)
    sv_mask = torch.from_numpy(sv_mask).to(device)
    
    output_tensor = MODEL(pv_points, pv_feattures, pv_mask, sv_points, sv_features, sv_mask)
    output = output_tensor.cpu().detach().numpy()
    return [output]

with Triton(config=TritonConfig(http_port=9000, grpc_port=9001, metrics_port=9002,log_verbose=0)) as triton:
    logger.info("loading model...")
    triton.bind(
        model_name = "pnet_pytorch",
        infer_func = __infer_fn,
        inputs = [
            Tensor(dtype=np.float32, shape=[2,-1]), 
            Tensor(dtype=np.float32, shape=[20,-1]), 
            Tensor(dtype=np.float32, shape=[1,-1]), 
            Tensor(dtype=np.float32, shape=[2,-1]),
            Tensor(dtype=np.float32, shape=[11,-1]),
            Tensor(dtype=np.float32, shape=[1,-1])
        ],
        outputs = [Tensor(dtype=np.float32, shape=[8])],
        config=ModelConfig(max_batch_size=1000),
        strict=True)
    logger.info("model loaded")
    triton.serve()
