from __future__ import division
import tensorflow as tf
import numpy as np


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


def bce_dice_loss(y_true, y_pred):
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


def bce_jacard_loss(y_true, y_pred):
    return jaccard(y_true, y_pred) + tf.keras.losses.binary_crossentropy(y_true, y_pred)