import torch
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV1SE, MiniFASNetV2, MiniFASNetV2SE
from src.utility import get_kernel, parse_model_name, parse_model_name_new_format
import os
import cv2
from src.data_io import transform as trans
import torch.nn.functional as F
# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
import onnx
from onnx_tf.backend import prepare
import time
# from pytorch2keras import pytorch_to_keras

from torch.autograd import Variable
from pytorch2keras import pytorch_to_keras
import numpy as np
import tensorflow as tf

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}

class AntiSpoofPredict():
    def __init__(self, device_id, model_path):
        self.device = torch.device('cuda:{}'.format(device_id)
                                    if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self._load_model(model_path)
        self.model.eval()

        print("init model successful")
    
    def _load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input, model_type, _ = parse_model_name_new_format(model_name)

        self.kernel_size = get_kernel(h_input, w_input)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                # if "model." == name_key[:6]:
                #     name_key = name_key[6:]
                new_state_dict[name_key] = value
            # print(new_state_dict)
            self.model.load_state_dict(new_state_dict, strict=False)
        else:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                # print(key)
                name_key = key
                if "model." == name_key[:6]:
                    name_key = name_key[6:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict, strict=False)
        return None

    def transform_input(self, img):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(mean, std)
        ])
        img = test_transform(img)
        # img = img.unsqueeze(0).to(self.device)
        img = img.unsqueeze(0)
        return img

    def predict(self, img):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        test_transform = trans.Compose([
            trans.ToTensor(),
            trans.Normalize(mean, std)
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            start = time.time()
            result = self.model.forward(img)
            print("time of forwarding ", time.time() - start)
            print("result", result)
            result = F.softmax(result, dim=1).cpu().numpy()
        return result

    def predict_non_normalize(self, img):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        test_transform = trans.Compose([
            trans.ToTensor(),
            # trans.Normalize(rgb_mean, rgb_std)
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            start = time.time()
            result = self.model.forward(img)
            print("time of forwarding ", time.time() - start)
            result = F.softmax(result, dim=1).cpu().numpy()
        return result



if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    device_id = 0
    model_path = "./resources/anti_spoof_models/2020-09-28-13-11_Anti_Spoofing_1.2_112x112_model_iter-150.pth"
    anti_model = AntiSpoofPredict(device_id, model_path)

    dummy_img = cv2.imread("/home/anonymous/Silent-Face-Anti-Spoofing/datasets/RGB_Images/1.2_112x112/test_caffee_model/0/1599816415801_18.png")
    dummy_input = anti_model.transform_input(dummy_img)

    input_np = np.random.uniform(0, 1, (1, 3, 112, 112))
    input_var = Variable(torch.FloatTensor(input_np))
    # input_var = torch.rand(1, 3, 112, 112) 

    k_model = pytorch_to_keras(anti_model.model, dummy_input, input_shapes=[(3, 112, 112,)], change_ordering=True, verbose=True, name_policy='short')
    # print("done")
    keras_file = 'keras_model.h5'
    tf.keras.models.save_model(k_model, keras_file)
    print("model saved")
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    # print(k_model)


    # import numpy as np
    # import torch
    # from pytorch2keras.converter import pytorch_to_keras
    # from torch.autograd import Variable
    # import tensorflow as tf
    # from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


    # # # Create and load model
    # # model = Model()
    # # model.load_state_dict(torch.load('model-checkpoint.pth'))
    # # model.eval()
    # model = anti_model.model

    # # Make dummy variables (and checking if the model works)
    # input_np = np.random.uniform(0, 1, (1, 3, 112, 112))
    # input_var = Variable(torch.FloatTensor(input_np))
    # output = model(input_var)

    # # Convert the model!
    # k_model = \
    #     pytorch_to_keras(model, input_var, (3, 112, 112), 
    #                     verbose=True, name_policy='short',
    #                     change_ordering=True)

    # # Save model to SavedModel format
    # tf.saved_model.save(k_model, "./models")

    # # Convert Keras model to ConcreteFunction
    # full_model = tf.function(lambda x: k_model(x))
    # full_model = full_model.get_concrete_function(
    #     tf.TensorSpec(k_model.inputs[0].shape, k_model.inputs[0].dtype))

    # # Get frozen ConcreteFunction
    # frozen_func = convert_variables_to_constants_v2(full_model)
    # frozen_func.graph.as_graph_def()

    # print("-" * 50)
    # print("Frozen model layers: ")
    # for layer in [op.name for op in frozen_func.graph.get_operations()]:
    #     print(layer)

    # print("-" * 50)
    # print("Frozen model inputs: ")
    # print(frozen_func.inputs)
    # print("Frozen model outputs: ")
    # print(frozen_func.outputs)

    # # Save frozen graph from frozen ConcreteFunction to hard drive
    # tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
    #                 logdir="./frozen_models",
    #                 name="frozen_graph.pb",
    #                 as_text=False)