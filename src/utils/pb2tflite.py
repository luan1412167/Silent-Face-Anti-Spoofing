# import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
# # make a converter object from the saved tensorflow file
# converter = tf.lite.TFLiteConverter.from_frozen_graph('resources/converted_models/tfmodel_worked_prep.pb', #TensorFlow freezegraph .pb model file
#                                                       input_arrays=['test_input'], # name of input arrays as defined in torch.onnx.export function before.
#                                                       output_arrays=['test_output']  # name of output arrays defined in torch.onnx.export function before.
#                                                       )
# # tell converter which type of optimization techniques to use
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# # to view the best option for optimization read documentation of tflite about optimization
# # go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

# # convert the model 
# tf_lite_model = converter.convert()
# # save the converted model 
# open('resources/converted_models/tflite_model.tflite', 'wb').write(tf_lite_model)

import tensorflow as tf
# make a converter object from the saved tensorflow file
converter = tf.lite.TFLiteConverter.from_frozen_graph('resources/converted_models/tfmodel_worked_prep.pb', #TensorFlow freezegraph .pb model file
                                                      input_arrays=['test_input'], # name of input arrays as defined in torch.onnx.export function before.
                                                      output_arrays=['test_output']  # name of output arrays defined in torch.onnx.export function before.
                                                      )
# tell converter which type of optimization techniques to use
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

# convert the model 
tf_lite_model = converter.convert()
# save the converted model 
open('resources/converted_models/model.tflite', 'wb').write(tf_lite_model)