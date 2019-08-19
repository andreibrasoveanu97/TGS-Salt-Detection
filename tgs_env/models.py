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




def build_model_1layer(input_layer, start_neurons, set_dropout=True):
    # 128 -> 64
    conv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    if set_dropout:
        pool1 = tf.keras.layers.Dropout(0.25)(pool1)

    # 64 -> 32
    conv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    if set_dropout:
        pool2 = tf.keras.layers.Dropout(0.5)(pool2)

    # 32 -> 16
    conv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    if set_dropout:
        pool3 = tf.keras.layers.Dropout(0.5)(pool3)

    # 16 -> 8
    conv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    if set_dropout:
        pool4 = tf.keras.layers.Dropout(0.5)(pool4)

    # Middle
    convm = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)

    # 8 -> 16
    deconv4 = tf.keras.layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = tf.keras.layers.concatenate([deconv4, conv4])
    if set_dropout:
        uconv4 = tf.keras.layers.Dropout(0.5)(uconv4)
    uconv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

    # 16 -> 32
    deconv3 = tf.keras.layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    uconv3 = tf.keras.layers.concatenate([deconv3, conv3])
    if set_dropout:
        uconv3 = tf.keras.layers.Dropout(0.5)(uconv3)
    uconv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

    # 32 -> 64
    deconv2 = tf.keras.layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2])
    if set_dropout:
        uconv2 = tf.keras.layers.Dropout(0.5)(uconv2)
    uconv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

    # 64 -> 128
    deconv1 = tf.keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1])
    if set_dropout:
        uconv1 = tf.keras.layers.Dropout(0.5)(uconv1)
    uconv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

    if set_dropout:
        uconv1 = tf.keras.layers.Dropout(0.5)(uconv1)
    output_layer = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

    return output_layer


def batchActivate(x):
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = tf.keras.layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = batchActivate(x)
    return x


def residual_block(blockInput, num_filters=16, batch_activate = False):
    x = batchActivate(blockInput)
    x = convolution_block(x, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = tf.keras.layers.Add()([x, blockInput])
    if batch_activate:
        x = batchActivate(x)
    return x


def build_model_resnet(input_layer, start_neurons, DropoutRatio = 0.5):
    conv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    pool1 = tf.keras.layers.Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    pool2 = tf.keras.layers.Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    pool3 = tf.keras.layers.Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(pool3)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8, True)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    pool4 = tf.keras.layers.Dropout(DropoutRatio)(pool4)

    # Middle
    convm = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(pool4)
    convm = residual_block(convm, start_neurons * 16)
    convm = residual_block(convm, start_neurons * 16, True)

    # 6 -> 12
    deconv4 = tf.keras.layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = tf.keras.layers.concatenate([deconv4, conv4])
    uconv4 = tf.keras.layers.Dropout(DropoutRatio)(uconv4)

    uconv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = residual_block(uconv4, start_neurons * 8, True)

    # 12 -> 25
    # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = tf.keras.layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = tf.keras.layers.concatenate([deconv3, conv3])
    uconv3 = tf.keras.layers.Dropout(DropoutRatio)(uconv3)

    uconv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)

    # 25 -> 50
    deconv2 = tf.keras.layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2])

    uconv2 = tf.keras.layers.Dropout(DropoutRatio)(uconv2)
    uconv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)

    # 50 -> 101
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = tf.keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1])

    uconv1 = tf.keras.layers.Dropout(DropoutRatio)(uconv1)
    uconv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)

    # uconv1 = Dropout(DropoutRatio/2)(uconv1)
    # output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = tf.keras.layers.Activation('sigmoid')(output_layer_noActi)

    return output_layer

if __name__ == '__main__':
    model = UnetModel()
    img = cv2.imread('./tgs/train/images/0a7e067255.png', cv2.IMREAD_GRAYSCALE)
    out_img = model(np.reshape(cv2.resize(img, (128, 128)), (1, 128, 128, 1)).astype(np.float64))
    out_img = np.reshape(out_img, (128, 128))
    cv2.imshow('mask', out_img)
    cv2.imshow('img', cv2.resize(img, (128, 128)))

    cv2.waitKey()
