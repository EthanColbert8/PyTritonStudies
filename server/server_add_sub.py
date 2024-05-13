import logging

import numpy as np

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

logger = logging.getLogger("examples.add_sub_python.server")
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


@batch
def _add_sub(**inputs):
    a_batch, b_batch = inputs.values()
    add_batch = a_batch + b_batch
    sub_batch = a_batch - b_batch
    return {"add": add_batch, "sub": sub_batch}

with Triton(config=TritonConfig(http_port=9000, grpc_port=9001, metrics_port=9002,log_verbose=0)) as triton:
    logger.info("Loading AddSub model")
    triton.bind(
        model_name="AddSub",
        infer_func=_add_sub,
        inputs=[
            Tensor(dtype=np.float32, shape=(-1,)),
            Tensor(dtype=np.float32, shape=(-1,)),
        ],
        outputs=[
            Tensor(name="add", dtype=np.float32, shape=(-1,)),
            Tensor(name="sub", dtype=np.float32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=128),
        strict=True,
    )
    logger.info("Serving model")
    triton.serve()
