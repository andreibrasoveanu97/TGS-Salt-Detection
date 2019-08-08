import tensorflow as tf
from keras import backend as K


def dice_coef(y_true, y_pred):
    intersection = K.sum(y_true.flatten() * y_pred.flatten(), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + 1) / (union + 1), axis=0)

