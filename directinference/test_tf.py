import tensorflow as tf
import numpy as np

gfile = "../../sonic-models/models/deepmet/1/model.graphdef"

f = tf.io.gfile.GFile(gfile, 'rb')
gdef = tf.compat.v1.GraphDef()
gdef.ParseFromString(f.read())
tf.import_graph_def(gdef)
sess = tf.compat.v1.Session()

output_tensor = sess.graph.get_tensor_by_name("output/BiasAdd:0")

a = np.random.rand(1,4500,8)
b = np.zeros((1,4500,1))
c = np.zeros((1,4500,1))
d = np.zeros((1,4500,1))

#with sess.as_default():
#    output = output_tensor.eval(feed_dict={"input:0": a, "input_cat0:0": b, "input_cat1:0": c, "input_cat2:0": d})
i = 0
while i < 10000:
    i += 1
    output = sess.run(output_tensor, feed_dict={"input:0": a, "input_cat0:0": b, "input_cat1:0": c, "input_cat2:0": d})
print(output.shape)
print(output)