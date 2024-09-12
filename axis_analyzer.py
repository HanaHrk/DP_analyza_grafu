import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# import imutils

# Advantage of using LAB space
# LAB has 3 channels just like RGB. But only 2 channels have color information (A and B),
# while L channel represents brightness value. Unlike RGB where we have to analyze all three channels,
# using LAB we can analyze only 2 channels. The benefit will be apparent when one has to analyze a large
# number of images.

"""
Method returns number containing information about collorfull 

How does it work?
1) obtain A and B channels of the image
2) find the mean value of the difference between them
3) determine a threshold above which all images can be labelled as color.
"""
def detect_image_colorfulness(image) -> float:
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # split the channels
    l, a, b = cv2.split(image_lab)

    # obtain difference between A and B channel at every pixel location
    de = abs(a - b)

    figure, axis = plt.subplots(1, 3)
    axis[0].imshow(image)
    axis[1].imshow(a)
    axis[2].imshow(b)
    plt.show()

    # find the mean of this difference
    mean = np.mean(de)
    return mean


# image = cv2.imread("/mnt/data/Corpora/chart_dataset/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/images/vertical_bar/PMC2664148___ddp05504.jpg", cv2.IMREAD_COLOR)
# image = cv2.imread("/mnt/data/Corpora/chart_dataset/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/images/horizontal_bar/PMC3104217___1471-2458-11-S4-S4-1.jpg", cv2.IMREAD_COLOR)
# image = cv2.imread("/mnt/data/Corpora/chart_dataset/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/images/vertical_bar/PMC2659770___pgen.1000449.g003.jpg", cv2.IMREAD_COLOR)

def analyze_imgs(img_folder):
    for file in os.listdir(img_folder):
        image = cv2.imread(os.path.join(img_folder, file), cv2.IMREAD_COLOR)
        colorness: float = detect_image_colorfulness(image)
        # set a threshold
        threshold: float = 3.0
        print(colorness)
        if colorness > threshold:
            print('its a color image...')
        else:
            print('Black and white image...')
        cv2.putText(image, "{:.2f}".format(colorness), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

        cv2.imshow('im', image)
        cv2.waitKey(0)

img_folder = "data/black_and_white_imgs"
analyze_imgs(img_folder)
img_folder = "data/colored_imgs"
analyze_imgs(img_folder)
