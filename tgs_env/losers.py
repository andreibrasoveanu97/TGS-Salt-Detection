from __future__ import division
import tensorflow as tf
import numpy as np


# def arith_or(array1, array2):
#     res = 0.0
#     for a, b in zip(array1, array2):
#         if a == 1.0 or b == 1.0:
#             res += 1.0
#         else:
#             res += 0.0
#
#     return res
#
#
# def arith_and(array1, array2):
#     res = 0.0
#     for a, b in zip(array1, array2):
#         if a == 1.0 and b == 1.0:
#             res += 1.0
#         else:
#             res += 0.0
#
#     return res
#
#
# def dice_loss(y_true, y_pred):
#     y_true_f = np.ravel(y_true)
#     y_pred_f = np.ravel(y_pred)
#     intersection = arith_and(y_true_f, y_pred_f)
#     union = arith_or(y_true_f, y_pred_f)
#     score = (2.0*(intersection + 1e-6) / (union + 1e-6))
#
#     return 1 - score


def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def combine_loss(y_true, y_pred):
    return 0.85*dice_loss(y_true, y_pred) + 0.15*tf.keras.losses.binary_crossentropy(y_true, y_pred)


def jaccard(y_true, y_pred):
    smooth = 1.

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth - intersection)
    return score


def jaccard_loss(y_true, y_pred):
    return 1 - jaccard(y_true, y_pred)