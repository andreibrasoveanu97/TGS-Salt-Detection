from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import cv2
# import tensorflow.contrib.eager as tfe
import numpy as np
from layers import UnetDecodeLayer, UnetMiddleLayer, UnetEncodeLayer


class UnetModel(tf.keras.Model):
    def __init__(self):
        super(UnetModel, self).__init__()
        kernel_shape = (3, 3)
        filters = 64
        # block1 encoder
        # input_shape = input_shape; ex. (128, 128)
        # output_shape = (input_shape) / 2; ex. (64, 64, filters)
        self.block1_encoder = UnetEncodeLayer(kernel_shape, filters)

        # block2 encoder
        # input_shape = output_shape
        # output_shape = (input_shape) / 2; ex. (32, 32, filters * 2)
        self.block2_encoder = UnetEncodeLayer(kernel_shape, filters * 2)

        # block 3 encoder
        # input_shape = output_shape
        # output_shape = (input_shape) / 2; ex. (16, 16, filters * 4)
        self.block3_encoder = UnetEncodeLayer(kernel_shape, filters * 4)

        # block 4 encoder
        # input_shape = output_shape
        # output_shape = (input_shape) / 2; ex. (8, 8, filters * 8)
        self.block4_encoder = UnetEncodeLayer(kernel_shape, filters * 8)

        # block middle
        # input_shape = output_shape
        # output_shape = (input_shape); ex. (8, 8, filters * 16)
        self.middle_block = UnetMiddleLayer(kernel_shape, filters * 16)

        # block 4 decoder
        # input_shape = output_shape
        # output_shape = (input_shape) * 2; ex. (16, 16, filters * 8)
        self.block4_decoder = UnetDecodeLayer(kernel_shape, filters * 8)

        # block 3 decoder
        # input_shape = output_shape
        # output_shape = (input_shape) * 2; ex. (32, 32, filters * 4)
        self.block3_decoder = UnetDecodeLayer(kernel_shape, filters * 4)

        # block 2 decoder
        # input_shape = output_shape
        # output_shape = (input_shape) * 2; ex. (64, 64, filters * 2)
        self.block2_decoder = UnetDecodeLayer(kernel_shape, filters * 2)

        # block 1 decoder
        # input_shape = output_shape
        # output_shape = (input_shape) * 2; ex. (128, 128, filters)
        self.block1_decoder = UnetDecodeLayer(kernel_shape, filters)

        self.out_layer = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")

    def call(self, inputs):
        block1_en_out, block1_en_out_pool = self.block1_encoder(inputs)
        block2_en_out, block2_en_out_pool = self.block2_encoder(block1_en_out_pool)
        block3_en_out, block3_en_out_pool = self.block3_encoder(block2_en_out_pool)
        block4_en_out, block4_en_out_pool = self.block4_encoder(block3_en_out_pool)

        middle_out = self.middle_block(block4_en_out_pool)

        block4_de_out = self.block4_decoder(middle_out, block4_en_out)
        block3_de_out = self.block3_decoder(block4_de_out, block3_en_out)
        block2_de_out = self.block2_decoder(block3_de_out, block2_en_out)
        block1_de_out = self.block1_decoder(block2_de_out, block1_en_out)
        return self.out_layer(block1_de_out)


def build_model(input_layer, start_neurons, set_dropout=True):
    # 128 -> 64
    conv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    if set_dropout:
        pool1 = tf.keras.layers.Dropout(0.25)(pool1)

    # 64 -> 32
    conv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    if set_dropout:
        pool2 = tf.keras.layers.Dropout(0.5)(pool2)

    # 32 -> 16
    conv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    if set_dropout:
        pool3 = tf.keras.layers.Dropout(0.5)(pool3)

    # 16 -> 8
    conv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    if set_dropout:
        pool4 = tf.keras.layers.Dropout(0.5)(pool4)

    # Middle
    convm = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    convm = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

    # 8 -> 16
    deconv4 = tf.keras.layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = tf.keras.layers.concatenate([deconv4, conv4])
    if set_dropout:
        uconv4 = tf.keras.layers.Dropout(0.5)(uconv4)
    uconv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    uconv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    # 16 -> 32
    deconv3 = tf.keras.layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = tf.keras.layers.concatenate([deconv3, conv3])
    if set_dropout:
        uconv3 = tf.keras.layers.Dropout(0.5)(uconv3)
    uconv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    uconv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    # 32 -> 64
    deconv2 = tf.keras.layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2])
    if set_dropout:
        uconv2 = tf.keras.layers.Dropout(0.5)(uconv2)
    uconv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    uconv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    # 64 -> 128
    deconv1 = tf.keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1])
    if set_dropout:
        uconv1 = tf.keras.layers.Dropout(0.5)(uconv1)
    uconv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    uconv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    if set_dropout:
        uconv1 = tf.keras.layers.Dropout(0.5)(uconv1)
    output_layer = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    return output_layer


if __name__ == '__main__':
    model = UnetModel()
    img = cv2.imread('./tgs/train/images/0a7e067255.png', cv2.IMREAD_GRAYSCALE)
    out_img = model(np.reshape(cv2.resize(img, (128, 128)), (1, 128, 128, 1)).astype(np.float64))
    out_img = np.reshape(out_img, (128, 128))
    cv2.imshow('mask', out_img)
    cv2.imshow('img', cv2.resize(img, (128, 128)))

    cv2.waitKey()
