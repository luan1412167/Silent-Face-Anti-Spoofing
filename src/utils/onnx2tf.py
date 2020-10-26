import numpy as np

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import onnx
from onnx_tf.backend import prepare
from src.data_io import transform as trans
import cv2
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def transform_input(img):
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

if __name__ == "__main__":
    # Load ONNX model and convert to TensorFlow format
    model_onnx = onnx.load('./resources/converted_models/model_new.onnx')
    dummy_img = cv2.imread("./datasets/RGB_Images/1.2_112x112/test_caffee_model/1/1599816472827_3.png")
    tf_rep = prepare(model_onnx, strict=False)
    dummy_input = transform_input(dummy_img)
    # output = prepare(model_onnx).run(dummy_input)

    # Export model as .pb file
    # tf_rep.export_graph('./resources/converted_models/tfmodel.pb')
    # print(pb_path)
    file = open('./resources/converted_models/tfmodel.pb', "wb")
    file.write(tf_rep.graph.as_graph_def().SerializeToString())
    file.close()