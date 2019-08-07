import tensorflow as tf


def dice_coef(y_true, y_pred):
    y_true = tf.reshape(y_true, -1)
    y_pred = tf.reshape(y_pred, -1)
    intersection = tf.reduce_prod(y_pred, y_true)
    return (2 * intersection + tf.keras.backend.epsilon) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + tf.keras.backend.epsilon)

