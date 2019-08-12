from __future__ import division
import tensorflow as tf
import numpy as np


def arith_or(array1, array2):
    res = []
    for a, b in zip(array1, array2):
        if a == 1.0 or b == 1.0:
            res.append(1.0)
        else:
            res.append(0.0)

    return res


def arith_and(array1, array2):
    res = []
    for a, b in zip(array1, array2):
        if a == 1.0 and b == 1.0:
            res.append(1.0)
        else:
            res.append(0.0)

    return res


def dice_loss(y_true, y_pred):
    y_true_f = np.ravel(y_true)
    y_pred_f = np.ravel(y_pred)
    intersection = sum(map(float, arith_and(y_true_f, y_pred_f)))
    union = sum(map(float, arith_or(y_true_f, y_pred_f)))
    score = (2.0*(intersection + 1e-6) / (union + 1e-6))

    return 1 - score


# def dice_loss(y_true, y_pred):
#     y_true_f = k.flatten(y_true)
#     y_pred_f = k.flatten(y_pred)
#     intersection = y_true_f * y_pred_f
#     union = y_true_f or y_pred_f
#     score = union / (2 * intersection)
#     return 1 - score


def combine_loss(y_true, y_pred):
    return 0.85*dice_loss(y_true, y_pred) + 0.15*tf.keras.losses.binary_crossentropy(y_true, y_pred)
