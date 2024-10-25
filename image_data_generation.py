import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import cv2

# Vytvoření vlastního generátoru pro načítání obrázků a masek
class ImageMaskGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, image_size, colorMode=0):
        self.image_filenames = sorted(os.listdir(image_dir))
        self.mask_filenames = sorted(os.listdir(mask_dir))
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.color_mode = colorMode

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_images = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        masks = []

        for i, m in zip(batch_images, batch_masks):
            if self.color_mode == 0:
                img = cv2.imread(os.path.join(self.image_dir, i), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(os.path.join(self.image_dir, i), cv2.IMREAD_COLOR)
            # resize image
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            img = img / 255

            mask = cv2.imread(os.path.join(self.mask_dir, m), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

            # mask = mask / 255
            mask[mask <= 250] = 0.0  # --> we must ensure that mask is binary image
            mask[mask > 250] = 1.0  # because we do a sigmoid on the last activation , we make sure that the correct labels are used

            # mask = np.expand_dims(mask, axis=-1)  # Přidání kanálu pro masku
            mask = np.reshape(mask, mask.shape + (1,))

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)


# Cesta ke složkám s obrázky a maskami
# image_dir = 'path/to/dataset/images/'
# mask_dir = 'path/to/dataset/masks/'


