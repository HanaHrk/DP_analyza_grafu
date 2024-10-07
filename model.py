from keras.models import *
from keras.layers import *
import tensorflow as tf
from tensorflow.keras.optimizers import *
from keras import backend as K


def f1(true, pred):

    print(true)

    # for metrics include these two lines, for loss, don't include them
    # these are meant to round 'pred' to exactly zeros and ones
    predLabels = K.argmax(pred, axis=-1)
    pred = K.one_hot(predLabels, 1)

    ground_positives = K.sum(true, axis=0)  # = TP + FN
    pred_positives = K.sum(pred, axis=0)  # = TP + FP
    true_positives = K.sum(true * pred, axis=0)  # = TP
    # all with shape (4,)

    precision = (true_positives + K.epsilon()) / (pred_positives + K.epsilon())
    recall = (true_positives + K.epsilon()) / (ground_positives + K.epsilon())
    # both = 1 if ground_positives == 0 or pred_positives == 0
    # shape (4,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    # not sure if this last epsilon is necessary
    # matematically not, but maybe to avoid computational instability
    # still with shape (4,)

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
    weighted_f1 = K.sum(weighted_f1)

    return weighted_f1  # for metrics, return only 'weighted_f1'


def jaccard_index(y_true, y_pred):

    # for metrics include these two lines, for loss, don't include them
    # these are meant to round 'pred' to exactly zeros and ones
    predLabels = K.argmax(pred, axis=-1)
    pred = K.one_hot(predLabels, 1)

    ground_positives = K.sum(true, axis=0)  # = TP + FN
    pred_positives = K.sum(pred, axis=0)  # = TP + FP
    true_positives = K.sum(true * pred, axis=0)  # = TP
    # all with shape (4,)

    precision = (true_positives + K.epsilon()) / (pred_positives + K.epsilon())
    recall = (true_positives + K.epsilon()) / (ground_positives + K.epsilon())
    # both = 1 if ground_positives == 0 or pred_positives == 0
    # shape (4,)

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    # not sure if this last epsilon is necessary
    # matematically not, but maybe to avoid computational instability
    # still with shape (4,)

    weighted_f1 = f1 * ground_positives / K.sum(ground_positives)
    weighted_f1 = K.sum(weighted_f1)

    # Intersection over Union -- Jaccard index
    # intersection for class 1 = tp1
    iou_foreground = tp_foreground / (tp_foreground + fp_foreground + fn_foreground)
    iou_background = tp_background / (tp_background + fp_background + fn_background)
    iou = (iou_foreground + iou_background) / 2
    return iou




def unet(pretrained_weights=None, input_size=(None, None, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs, conv10)
    model.summary()

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy', fg_iou, bg_iou, iou]) # , tf.keras.metrics.MeanIoU(num_classes = 1)

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def fg_iou(y_true, y_pred):
    th = K.cast(K.greater(y_pred, 0.5), y_true.dtype)
    intersection = K.sum(th * y_true)
    union = K.sum(K.clip(th + y_true, 0, 1))
    return intersection / (union + K.epsilon())

def bg_iou(y_true, y_pred):
    th = K.cast(K.less_equal(y_pred, 0.5), y_true.dtype)
    y_true_inv = K.cast(K.less_equal(y_true, 0.5), y_true.dtype)
    intersection = K.sum(th * y_true_inv)
    union = K.sum(K.clip(th + y_true_inv, 0, 1))
    return intersection / (union + K.epsilon())

def iou(y_true, y_pred):
    iou_1 = fg_iou(y_true, y_pred)
    iou_2 = bg_iou(y_true, y_pred)
    #if tf.debugging.is_nan(iou_1):
    #    return iou_2
    #if tf.debugging.is_nan(iou_2):
    #    return iou_1
    return (iou_1 + iou_2) / 2


