import logging

import numpy as np
import tensorflow as tf  # pytype: disable=import-error

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig

logger = logging.getLogger("examples.deepmet.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

def _load_model():
    gfile = "../../sonic-models/models/deepmet/1/model.graphdef"
    f = tf.io.gfile.GFile(gfile, 'rb')
    gdef = tf.compat.v1.GraphDef()
    gdef.ParseFromString(f.read())
    tf.import_graph_def(gdef)
    sess = tf.compat.v1.Session()
    output_tensor = sess.graph.get_tensor_by_name("output/BiasAdd:0")
    
    return sess, output_tensor

SESS, OUTPUT_TENSOR = _load_model()

@batch
def __infer_fn(**inputs):
    tensor, tensor_cat0, tensor_cat1, tensor_cat2 = inputs.values()
    #tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)
    #tensor_cat0 = tf.convert_to_tensor(tensor_cat0, dtype=tf.float32)
    #tensor_cat1 = tf.convert_to_tensor(tensor_cat1, dtype=tf.float32)
    #tensor_cat2 = tf.convert_to_tensor(tensor_cat2, dtype=tf.float32)
    output = SESS.run(OUTPUT_TENSOR, feed_dict={"input:0": tensor, "input_cat0:0": tensor_cat0, "input_cat1:0": tensor_cat1, "input_cat2:0": tensor_cat2})
    return [output]

with Triton(config=TritonConfig(http_port=9000, grpc_port=9001, metrics_port=9002,log_verbose=0)) as triton:
    logger.info("loading model...")
    triton.bind(
        model_name = "mlp_random_tensorflow_graphdef",
        infer_func = __infer_fn,
        inputs = [
            Tensor(dtype=np.float32, shape=[4500, 8]), 
            Tensor(dtype=np.float32, shape=[4500, 1]), 
            Tensor(dtype=np.float32, shape=[4500, 1]), 
            Tensor(dtype=np.float32, shape=[4500, 1])
        ],
        outputs = [Tensor(name="output/BiasAdd", dtype=np.float32, shape=[2])],
        config=ModelConfig(max_batch_size=1000),
        strict=True)
    logger.info("model loaded")
    triton.serve()
