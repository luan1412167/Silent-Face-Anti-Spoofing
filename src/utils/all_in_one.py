from pytorch2onnx import AntiSpoofPredict
import cv2
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
from onnx_tf.backend import prepare
import onnx
import tensorflow as tf
# from tensorflow.contrib.lite.python import Interpreter

if __name__=="__main__":
    # device_id = 0
    # model_path = "./resources/anti_spoof_models/2020-09-19-04-15_Anti_Spoofing_1.2_112x112_model_iter-399.pth"
    # anti_model = AntiSpoofPredict(device_id, model_path)
    # dummy_img = cv2.imread("/home/dmp/Silent-Face-Anti-Spoofing/datasets/RGB_Images/1.2_112x112/test_caffee_model/0/1599816415801_18.png")
    # dummy_input = anti_model.transform_input(dummy_img)
    # # Export the model
    # print("dummy_input", dummy_input.shape)

    # torch_out = torch.onnx._export(anti_model.model,             # model being run
    #                           dummy_input,                       # model input (or a tuple for multiple inputs)
    #                           'resources/converted_models/newmodel.onnx', # where to save the model (can be a file or file-like object)
    #                           export_params=True,       # store the trained parameter weights inside the model file
    #                           input_names=['main_input'],  # specify the name of input layer in onnx model
    #                           output_names=['main_output'])     # specify the name of input layer in onnx model


    # model_onnx = onnx.load('resources/converted_models/newmodel.onnx')
    # # onnx.checker.check_model(model_onnx)

    # # prepare model for exporting to tensorFlow using tensorFlow backend
    # tf_rep = prepare(model_onnx, strict=True)
    # start_time = mills()
    # print(tf_rep.run(dummy_input))
    # # end_time = mills()
    # # print(end_time - start_time)
    # print(tf_rep.inputs) # Input nodes to the model
    # print('-----')
    # print(tf_rep.outputs) # Output nodes from the model
    # print('-----')
    # print(tf_rep.tensor_dict) # All nodes in the model

    # # # export tensorFlow backend to tensorflow tf file
    # print("onnx", tf_rep.run(dummy_input))
    # tf_rep.export_graph('resources/converted_models/newmodel.pb')

    # converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph('resources/converted_models/newmodel.pb', #TensorFlow freezegraph .pb model file
    #                                                   input_arrays=['main_input'], # name of input arrays as defined in torch.onnx.export function before.
    #                                                   output_arrays=['main_output']  # name of output arrays defined in torch.onnx.export function before.
    #                                                   )

    # # converter = tf.saved_model.load('resources/converted_models/newmodel.pb') #TensorFlow freezegraph .pb model file
    # # tf_tensor = tf.convert_to_tensor(dummy_input.numpy())
    # # print(tf_tensor.shape)
    # # print(converter(tf_tensor))
                                                      
    # # tell converter which type of optimization techniques to use
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # # to view the best option for optimization read documentation of tflite about optimization
    # # go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

    # # convert the model 
    # tf_lite_model = converter.convert()
    # # save the converted model 
    # open('resources/converted_models/newmodel.tflite', 'wb').write(tf_lite_model)

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="resources/converted_models/newmodel.tflite")
    input_details = interpreter.get_input_details()
    # print(interpreter.)
    output_details = interpreter.get_output_details()
    print(input_details, output_details)
    interpreter.allocate_tensors()

    # Get input and output tensors.

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    print(input_details)
    # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    input_data = dummy_input.numpy()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    start_time = mills()

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    end_time = mills()
    print("Time taken to one inference in milliseconds", end_time - start_time)
    print("output of model",output_data)