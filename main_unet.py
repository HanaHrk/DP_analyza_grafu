from image_data_generation import ImageMaskGenerator
from image_loader import load_images
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

import model as m
import cv2
import numpy as np

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

    BATCH_SIZE = 4
    IMAGE_SIZE = [512, 512]

    # Vytvoření generátoru
    # image_dir = 'path/to/dataset/images/'
    # mask_dir = 'path/to/dataset/masks/'
    train_image_dir = os.path.join(dataDir, "axis_analysis", "train_images")
    train_mask_dir = os.path.join(dataDir, "axis_analysis", "train_masks")
    train_generator = ImageMaskGenerator(train_image_dir, train_mask_dir, BATCH_SIZE, IMAGE_SIZE)

    val_image_dir = os.path.join(dataDir, "axis_analysis", "val_images")
    val_mask_dir = os.path.join(dataDir, "axis_analysis", "val_masks")
    val_generator = ImageMaskGenerator(val_image_dir, val_mask_dir, BATCH_SIZE, IMAGE_SIZE)


    # trainX, trainY = load_images(os.path.join(dataDir, "axis_analysis"), colorMode=colorMode)
    # testX, testY = load_images(os.path.join(dataDir, "axis_analysis"), colorMode=colorMode)
    # valX, valY = load_images(os.path.join(dataDir, "axis_analysis"), colorMode=colorMode)

    cfg = ConfigProto()
    cfg.gpu_options.allow_growth = True
    session = InteractiveSession(config=cfg)


    model = m.unet(input_size=(None, None, channels))
    model_checkpoint = ModelCheckpoint(model_path, monitor='loss', verbose=1, save_best_only=True)
    # es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)
    # model.fit(trainX, trainY, batch_size=3, epochs=200, callbacks=[model_checkpoint, es], validation_data=(valX, valY), shuffle=True) # callbacks=[model_checkpoint, es]
    # model.fit(trainX, trainY, batch_size=2, epochs=100, callbacks=[model_checkpoint], validation_data=(valX, valY), shuffle=True)  # callbacks=[model_checkpoint, es]
    model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=10, callbacks=[model_checkpoint], validation_data=val_generator, shuffle=True)

    size = IMAGE_SIZE
    for fn in os.listdir(os.path.join(dataDir, "axis_analysis", "sample_images")):
        img = cv2.imread(os.path.join(dataDir, "axis_analysis", "sample_images", fn), cv2.IMREAD_COLOR)
        print ("File name:", fn)
        print ("Img size:", img.shape)
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

    # for i in range(testX.shape[0]):
    #     patch = testX[i]
    #     patch = np.reshape(patch, (1, 240, 320, 3))
    #     pred = model.predict(patch)
    #     res = pred*255
    #     cv2.imshow("visu", res)
    #     cv2.waitKey()
