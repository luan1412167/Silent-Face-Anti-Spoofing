import numpy as np
import tensorflow as tf
from pytorch2onnx import AntiSpoofPredict
import cv2
import torch
import time

#load pytorch
device_id = 0
model_path = "./resources/anti_spoof_models/2020-09-28-13-11_Anti_Spoofing_1.2_112x112_model_iter-150.pth"
anti_model = AntiSpoofPredict(device_id, model_path)

dummy_img = cv2.imread("./datasets/RGB_Images/1.2_112x112/test_caffee_model/0/1599816416115_69.png")
dummy_output = anti_model.predict(dummy_img)
print("dummy_output_pytorch", dummy_output)


inputx = anti_model.transform_input(dummy_img)
inputx = inputx.permute(0, 2, 3, 1).numpy()
print(inputx.shape)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
print(input_details)
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
print(input_data.shape)
# input_data = dummy_input.numpy()
interpreter.set_tensor(input_details[0]['index'], inputx)
# start_time = mills()

interpreter.invoke()
start_time = time.time()
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
end_time = time.time()
# end_time = mills()
print("Time taken to one inference in milliseconds", end_time - start_time)
print("output of model",output_data)