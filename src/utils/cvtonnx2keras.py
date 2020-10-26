import onnx
from onnx2keras import onnx_to_keras
import torch
from torch.autograd import Variable
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# Load ONNX model
onnx_model = onnx.load('resources/converted_models/model.onnx')

# Call the converter (input - is the main model input name, can be different for your model)
k_model = onnx_to_keras(onnx_model, ['test_input'], change_ordering=True)

# input_np = np.random.uniform(0, 1, (1, 3, 112, 112))
# input_var = Variable(torch.FloatTensor(input_np))
# print(input_var.shape)
# # output = model(input_var)
# # pytorch_output = output.data.numpy()
# inputks = np.transpose(input_np, [0, 2, 3, 1])
# print(inputks.shape)
# keras_output = k_model.predict(inputks)
# print("keras_output", keras_output)