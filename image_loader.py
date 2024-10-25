import cv2
import sys
import random
import numpy as np
import os
import fnmatch
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator

'''  
0 - grayscale
1 - rgb
'''
def load_images(path, colorMode=0):
    imgs = []
    masks = []
    for fn in os.listdir(os.path.join(path, "train_images")):
        if colorMode == 0:
            img = cv2.imread(os.path.join(path, "train_images", fn), cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(os.path.join(path, "train_images", fn), cv2.IMREAD_COLOR)
        if img is None:
            print("Could not load image: ", os.path.join(path, "train_images", fn))
        if img is None: continue

        # resize image
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

        img = img / 255
        # img = np.reshape(img, img.shape + (1,))
        # print("Img shape:", img.shape)

        mask = cv2.imread(os.path.join(path, "train_masks", fn), cv2.IMREAD_GRAYSCALE)
        # resize  mask
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # mask = mask / 255
        mask[mask <= 250] = 0.0  # --> we must ensure that mask is binary image
        mask[mask > 250] = 1.0  # because we do a sigmoid on the last activation , we make sure that the correct labels are used

        mask = np.reshape(mask, mask.shape + (1,))
        # print ("Mask shape:", mask.shape)



        imgs.append(img.astype(np.float16))
        masks.append(mask.astype(np.float16))

    imgs = np.array(imgs, dtype=np.float16)
    masks = np.array(masks, np.float16)

    print("Data shapes:", imgs.shape, masks.shape)
    return (imgs, masks)

def myImageDataGenerator(path, batch_size, size, seed=10):
    image_datagen = ImageDataGenerator(rescale=1./255)
    mask_datagen = ImageDataGenerator(rescale=1./255)

    #print("Patch generator:", path, "size:", size)

    image_generator = image_datagen.flow_from_directory(
        path,
        classes = ["train_images"],
        color_mode='bgr',
        class_mode=None,
        target_size=(size[0], size[1]),
        batch_size=batch_size,
        seed = seed
    )
    mask_generator = mask_datagen.flow_from_directory(
        path,
        classes=["train_masks"],
        color_mode='grayscale',
        class_mode=None,
        target_size=(size[0], size[1]),
        batch_size=batch_size,
        seed = seed
    )
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

