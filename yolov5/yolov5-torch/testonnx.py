import onnx, onnxsim, onnxruntime
import cv2
import numpy as np

sess_options = onnxruntime.SessionOptions()
model = onnx.load('./yolov5s-sim.onnx')
# 'TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'
onnx_model = onnxruntime.InferenceSession(model.SerializeToString(), sess_options,
                                            providers=['CPUExecutionProvider'])

image = cv2.imread('./image.jpg', flags=cv2.IMREAD_COLOR)
image = cv2.resize(image, (640, 640), dst=None, fx=None, fy=None, interpolation=None)
image = image.transpose(2,0,1)
image = np.expand_dims(image, axis=0)
image = image.astype(np.float32)

mu = np.mean(image, axis=0)
sigma = np.std(image, axis=0)
image =  (image - mu) / (sigma+1e-8)
print(image.shape)
out = onnx_model.run(None, {onnx_model.get_inputs()[0].name: image})
np.set_printoptions(threshold=np.inf)
print(out[0].shape, out[0]>0)