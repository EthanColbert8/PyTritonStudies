import onnxruntime as rt
import numpy as np

path = "../../sonic-models/models/particlenet_AK4/1/model.onnx"

providers = [("ROCMExecutionProvider")]
sess_options = rt.SessionOptions()
sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = rt.InferenceSession(path, sess_options=sess_options, providers=['ROCMExecutionProvider', 'CPUExecutionProvider'])
print("provider: ", sess.get_providers())
input_name0 = sess.get_inputs()[0].name
input_name1 = sess.get_inputs()[1].name
input_name2 = sess.get_inputs()[2].name
input_name3 = sess.get_inputs()[3].name
input_name4 = sess.get_inputs()[4].name
input_name5 = sess.get_inputs()[5].name

output_name = sess.get_outputs()[0].name

nevts = 5000
pf_points   = np.random.randn(nevts, 2,  100).astype(np.float32)
pf_features = np.random.randn(nevts, 20, 100).astype(np.float32)
pf_mask     = np.random.randn(nevts, 1,  100).astype(np.float32)
sv_points   = np.random.randn(nevts, 2,  10).astype(np.float32)
sv_features = np.random.randn(nevts, 11, 10).astype(np.float32)
sv_mask     = np.random.randn(nevts, 1,  10).astype(np.float32)
#pf_points = np.zeros((nevts, 2, 100)).astype(np.float32)
#pf_features = np.zeros((nevts, 20, 100)).astype(np.float32)
#pf_mask = np.zeros((nevts, 1, 100)).astype(np.float32)
#sv_points = np.zeros((nevts, 2, 10)).astype(np.float32)
#sv_features = np.zeros((nevts, 11, 10)).astype(np.float32)
#sv_mask = np.zeros((nevts, 1, 10)).astype(np.float32)

pred_onx = sess.run([output_name], {input_name0: pf_points, input_name1: pf_features, input_name2: pf_mask, input_name3: sv_points, input_name4: sv_features, input_name5: sv_mask})[0]
