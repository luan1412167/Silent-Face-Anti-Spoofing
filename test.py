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

SAMPLE_IMAGE_PATH = "./images/sample/"


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        # print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return True
    else:
        return True


def test(image_folder, model_dir, device_id):
    image_paths = glob.glob(image_folder, "/*/*")
    for image_path in image_paths:
        model_test = AntiSpoofPredict(device_id)
        image_cropper = CropImage()
        image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
        result = check_image(image)
        if result is False:
            return
        image_bbox = model_test.get_bbox(image)
        prediction = np.zeros((1, 2))
        test_speed = 0
        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            model_out = model_test.predict(img, os.path.join(model_dir, model_name))
            prediction += model_out
            test_speed += time.time()-start

        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label]
        if label == 1:
            print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        print("Prediction cost {:.2f} s".format(test_speed))
        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            image,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    # format_ = os.path.splitext(image_name)[-1]
    # result_image_name = image_name.replace(format_, "_result" + format_)
    # cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)


def test_video(image_name, model_dir, device_id):
    cap = cv2.VideoCapture("/home/dmp/Videos/testdata/real/2020-09-11-133732.webm")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    while cap.isOpened():
        model_test = AntiSpoofPredict(device_id)
        image_cropper = CropImage()
        # image = cv2.imread(SAMPLE_IMAGE_PATH + image_name)
        ret, image = cap.read()
        result = check_image(image)
        if result is False:
            return
        image_bbox = model_test.get_bbox(image)
        prediction = np.zeros((1, 3))
        test_speed = 0
        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            # if model_name != "2020-09-11-10-17_Anti_Spoofing_1.2_112x112_model_iter-36.pth":
                # continue
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            # h_input, w_input, model_type, scale = parse_model_name_new_format(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            print(model_type, scale)
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            if h_input == 112:
                model_out = model_test.predict(img, os.path.join(model_dir, model_name))
            else:
                model_out = model_test.predict_non_normalize(img, os.path.join(model_dir, model_name))

            # if model_out.shape[1] == 2: 
            #     model_out = np.expand_dims(np.append(model_out[0], 0.33), axis=0)
            print(model_out)
            prediction += model_out

            test_speed += time.time()-start

        # draw result of prediction
        label = np.argmax(prediction)
        # print(prediction)
        score = prediction[0][label]
        if score < .95:
            label = 0
        if label == 1:
            print("Image '{}' is Real Face. Score: {:.2f}.".format("image_name", score))
            result_text = "RealFace Score: {:.2f}".format(score)
            color = (255, 0, 0)
        else:
            print("Image '{}' is Fake Face. Score: {:.2f}.".format("image_name", score))
            result_text = "FakeFace Score: {:.2f}".format(score)
            color = (0, 0, 255)
        print("Prediction cost {:.2f} s".format(test_speed))
        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            image,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)
        cv2.imshow("image", image)
        cv2.imshow("img", img)
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        # format_ = os.path.splitext(image_name)[-1]
        # result_image_name = image_name.replace(format_, "_result" + format_)
        # cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)

def parser_data(test_folder):
    image_paths = glob.glob(test_folder + "/*/*.png")
    datas = []
    for image_path in image_paths:
        target = image_path.split("/")[-2]
        datas.append([image_path, int(target)])
        # print(image_path, int(target))

    return datas 

def calculate_acc(test_folder, model_dir, device_id):

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    
    datas = parser_data(test_folder)
    print("num_of_data ", len(datas))
    idx = 0
    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    for model_name in os.listdir(model_dir):
        # if model_name != "2020-09-11-02-29_Anti_Spoofing_2.7_112x112_model_iter-9.pth":
        #     continue
        print(model_name)
        # h_input, w_input, model_type, scale = parse_model_name(model_name)

        h_input, w_input, model_type, scale = parse_model_name_new_format(model_name)

        # for threshold in np.arange(0.60, 0.9, 0.05):
        if True:
            pred, truth = [], []
            bcm = BinaryClassificationMeter()
            threshold = 0.5
            print("threshold ", threshold)
            random.seed(0)
            random.shuffle(datas)
            for img_path, target in datas:
                idx += 1
                image_name = os.path.basename(img_path)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                image_bbox = model_test.get_bbox(image)
                prediction = np.zeros((1, 2))
                test_speed = 0

                # param = {
                #     "org_img": image,
                #     "bbox": image_bbox,
                #     "scale": scale,
                #     "out_w": w_input,
                #     "out_h": h_input,
                #     "crop": False,
                # }
                # if scale is None:
                #     param["crop"] = False
                # img = image_cropper.crop(**param)
                # img = cv2.resize(image, (80,80))
                img = image


                start = time.time()
                model_out = model_test.predict(img, os.path.join(model_dir, model_name))
                # print(model_out)
                prediction += model_out
                test_speed += time.time()-start

                # draw result of prediction
                label = np.argmax(prediction)
                value = prediction[0][label]
                if value < threshold:
                    label = 0 
                if label == 1:
                    # print("Image '{}' is Real Face. Score: {:.2f}.".format(image_name, value))
                    result_text = "RealFace Score: {:.2f}".format(value)
                    color = (255, 0, 0)
                else:
                    # print("Image '{}' is Fake Face. Score: {:.2f}.".format(image_name, value))
                    result_text = "FakeFace Score: {:.2f}".format(value)
                    color = (0, 0, 255)
                # print("Prediction cost {:.2f} s".format(test_speed))
                cv2.rectangle(
                    image,
                    (image_bbox[0], image_bbox[1]),
                    (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                    color, 2)
                cv2.putText(
                    image,
                    result_text,
                    (image_bbox[0], image_bbox[1] - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)
                pred.append(label)
                truth.append(target)
                # print(idx)
                # cv2.imshow("image", image)
                # k = cv2.waitKey(500)
                # if k == ord("q"):
                #     break
                if idx%1000==0:
                    bcm.update(np.array(pred), np.array(truth))
                    print("PRE={}, REC={}, F1={}".format(bcm.pre, bcm.rec, bcm.f1))
                    break
            bcm.update(np.array(pred), np.array(truth))
            print("PRE={}, REC={}, F1={}".format(bcm.pre, bcm.rec, bcm.f1))
            bcm.reset()
                # format_ = os.path.splitext(image_name)[-1]
            # result_image_name = image_name.replace(format_, "_result" + format_)
            # cv2.imwrite(SAMPLE_IMAGE_PATH + result_image_name, image)


if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        # default="./important_models/Anti_Spoofing_1.2_112x112",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="20190911_001_daytime_front_190911093532_2_3.png",
        help="image used to test")
    args = parser.parse_args()
    # test(args.image_name, args.model_dir, args.device_id)

    test_video(args.image_name, args.model_dir, args.device_id)
    # calculate_acc("/home/dmp/Silent-Face-Anti-Spoofing/datasets/RGB_Images/1.2_112x112/test_caffee_model", args.model_dir, args.device_id)