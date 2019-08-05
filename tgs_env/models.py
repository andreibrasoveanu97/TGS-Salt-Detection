from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.compat.v1.enable_eager_execution()

class UnetModel(tf.keras.Model):