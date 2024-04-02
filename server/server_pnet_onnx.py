import logging

import numpy as np
import onnxruntime as rt

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

path = "/work1/yfeng/yfeng/sonic-models/models/particlenet_AK4/1/model.onnx"
sess = rt.InferenceSession(path)
input_name0 = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
input_name3 = sess.get_inputs()[3].name
input_name4 = sess.get_inputs()[4].name
input_name5 = sess.get_inputs()[5].name
output_name = sess.get_outputs()[0].name


logger = logging.getLogger("examples.pnet_onnx.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

@batch
def __infer_fn(**inputs):
    pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask = inputs.values()
    output = sess.run([output_name], {input_name0: pf_points, input_name1: pf_features, input_name2: pf_mask, input_name3: sv_points, input_name4: sv_features, input_name5: sv_mask})[0]
    return [output]

with Triton(config=TritonConfig(http_port=9000, grpc_port=9001, metrics_port=9002,log_verbose=0)) as triton:
    logger.info("loading model...")
    triton.bind(
        model_name = "pnet_onnx",
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
