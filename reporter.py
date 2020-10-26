# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
import glob

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name, BinaryClassificationMeter, parse_model_name_new_format
warnings.filterwarnings('ignore')
import random
import pandas as pd
from prettytable import PrettyTable

SAMPLE_IMAGE_PATH = "./images/sample/"

if __name__ == "__main__":

    model_dir = "./saved_logs/snapshot/Anti_Spoofing_1.2_112x112"
    model_paths = glob.glob(model_dir + "/*")
    model_name = "2020-09-19-06-18_Anti_Spoofing_1.2_112x112_model_iter-480.pth"
    model_iter = 0
    #report data
    df = pd.DataFrame(columns=['it', 'tp', 'tn', 'fp', 'fn', 'pr', 're', 'f1'])
    x = PrettyTable(['it', 'tp', 'tn', 'fp', 'fn', 'pr', 're', 'f1'])

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        print(model_name)
        model_iter = model_name.split(".")[1].split("-")[-1]
        model_path = os.path.join(model_dir, model_name)
        folder_path = "/home/anonymous/Videos/sanity_data"
        video_paths = glob.glob(folder_path + "/*/*")
        truth = 0

        # initialize evaluation methods 
        preds, truths = [], []
        bcm = BinaryClassificationMeter()

        h_input, w_input, model_type, scale = parse_model_name_new_format(model_name)
        model_test = AntiSpoofPredict(0, model_path)
        image_cropper = CropImage()

        confidence = .70

        for video_path in video_paths:
            # print(video_path)
            # if video_path != "/home/anonymous/Videos/sanity_data/real/2020-09-18-095512.webm":
            #     continue
            if "real" in video_path:
                truth = 1
        
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            # cv2.namedWindow("img", cv2.WINDOW_NORMAL)

            
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break

                image_bbox = model_test.get_bbox(image)
                if len(image_bbox)==0:
                    continue

                prediction = np.zeros((1, 2))
                test_speed = 0

                # sum the prediction from single model's result
                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                
                img = image_cropper.crop(**param)

                start = time.time()
                model_out = model_test.predict(img)

                prediction += model_out

                test_speed += time.time()-start

                # draw result of prediction
                label = np.argmax(prediction)
                score = prediction[0][label]
                if score < confidence:
                    label = 0
                preds.append(label)
                truths.append(truth)


                #interface
                # if label == 1:
                #     # print("Image '{}' is Real Face. Score: {:.2f}.".format("image_name", score))
                #     result_text = "RealFace Score: {:.2f}".format(score)
                #     color = (255, 0, 0)
                # else:
                #     # print("Image '{}' is Fake Face. Score: {:.2f}.".format("image_name", score))
                #     result_text = "FakeFace Score: {:.2f}".format(score)
                #     color = (0, 0, 255)
                # # print("Prediction cost {:.2f} s".format(test_speed))
                # cv2.rectangle(
                #     image,
                #     (image_bbox[0], image_bbox[1]),
                #     (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                #     color, 2)
                # cv2.putText(
                #     image,
                #     result_text,
                #     (image_bbox[0], image_bbox[1] - 5),
                #     cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)
                # cv2.imshow("image", image)
                # cv2.imshow("img", img)
                # k = cv2.waitKey(1)
                # if k == ord("q"):
                #     break
                bcm.update(np.array(preds), np.array(truths))
        
        report = {'it':model_iter, 'tp':bcm.tp.item(), 'tn':bcm.tn.item(), 'fp':bcm.fp.item(),
                  'fn':bcm.fn.item(), 'pr':bcm.pre.item(), 're':bcm.rec.item(), 'f1':bcm.f1.item()}
        df = df.append(report, ignore_index=True)
        print("pre={}, rec={}, f1={}, tp={}, tn={}, fp={}, fn={}"
              .format(bcm.pre, bcm.rec, bcm.f1, bcm.tp, bcm.tn, bcm.fp, bcm.fn))
        df.to_csv('/home/anonymous/Silent-Face-Anti-Spoofing/reporter.csv',
                   sep='\t', encoding='utf-8', header='true')

        # pretty table 
        x.add_row([model_iter, bcm.tp.item(), bcm.tn.item(), bcm.fp.item(),
                      bcm.fn.item(), bcm.pre.item(), bcm.rec.item(), bcm.f1.item()])
        data = x.get_string()
        with open('reporter.txt', 'w') as f:
            f.write(data)

