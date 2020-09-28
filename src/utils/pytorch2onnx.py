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


MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}

class AntiSpoofPredict():
    def __init__(self, device_id, model_path):
        # self.device = torch.device('cuda:{}'.format(device_id)
        #                             if torch.cuda.is_available() else 'cpu')
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
    device_id = 0
    model_path = "./resources/anti_spoof_models/2020-09-19-04-15_Anti_Spoofing_1.2_112x112_model_iter-399.pth"
    anti_model = AntiSpoofPredict(device_id, model_path)

    dummy_img = cv2.imread("/home/dmp/Silent-Face-Anti-Spoofing/datasets/RGB_Images/1.2_112x112/test_caffee_model/0/1599816415801_18.png")
    dummy_input = anti_model.transform_input(dummy_img)
    # test_speed = 0
    # for _ in range(1000):
    #     start = time.time()
    #     dummy_output = anti_model.predict(dummy_img)
    #     print("dummy_output", dummy_output)
    #     print(time.time()-start)
    #     test_speed += time.time()-start

    # print("test_speed", test_speed/1000)
    
    converted_model = "./resources/converted_models"
    torch.onnx.export(anti_model.model, dummy_input, converted_model + "/model_new.onnx",
                      input_names=['test_input'], output_names=['test_output'])

    # model_onnx = onnx.load('./resources/converted_models/model.onnx')

    # # tf_rep = prepare(model_onnx)
    # output = prepare(model_onnx).run(dummy_input)
    # print(output)