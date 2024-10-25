from image_data_generation import ImageMaskGenerator
from image_loader import load_images
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

import model as m
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


if __name__ == '__main__':
    dataDir = "data"
    colorMode = 0
    if len(sys.argv) > 3:
        model_path = sys.argv[3]
    else:
        model_path = "best_model.h5"

    channels = 1 if colorMode == 0 else 3

    BATCH_SIZE = 1
    IMAGE_SIZE = [512, 512]

    cfg = ConfigProto()
    cfg.gpu_options.allow_growth = True
    session = InteractiveSession(config=cfg)

    # model = m.unet(input_size=(None, None, channels))
    model = tf.keras.models.load_model("unet_axis_prediction_50_epochs_channels_1.h5")

    size = IMAGE_SIZE
    for fn in os.listdir(os.path.join("sample_images", "vertical_bars")):
        if channels == 1:
            img = cv2.imread(os.path.join("sample_images", "vertical_bars", fn), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(os.path.join("sample_images", "vertical_bars", fn), cv2.IMREAD_COLOR)
        print("File name:", fn)
        print("Img size:", img.shape)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = np.reshape(img, (1, size[0], size[1], channels))
        pred = model.predict(img)
        res = pred * 255
        print("res shape:", res.shape)
        res = np.reshape(res, (size[0], size[1]))
        res = res.astype(np.uint8)

        cv2.imwrite(os.path.join("keras_output", "prediction_"+fn), res)
        #cv2.imshow("visu", res)
        #cv2.waitKey()


