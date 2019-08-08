import tensorflow as tf
from keras import backend as K

def dice_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (K.sum(y_true_f) + K.sum(y_pred_f)) / (2. * K.sum(intersection))
    return score

def combine_loss(y_true, y_pred):
    return 0.85*dice_loss(y_true, y_pred) + 0.15*tf.keras.losses.binary_crossentropy(y_true, y_pred)