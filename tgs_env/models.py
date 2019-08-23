from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import cv2
# import tensorflow.contrib.eager as tfe
import numpy as np

from layers import UnetDecodeLayer, UnetMiddleLayer, UnetEncodeLayer

# tf.enable_eager_execution()


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


def squeeze_excitation_layer(_input, channels, ratio=0.25):
    a = channels / ratio
    x = tf.keras.layers.GlobalAveragePooling2D()(_input)
    x = tf.keras.layers.Dense(a)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dense(_input.shape[-1])(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    return tf.keras.layers.multiply([x, _input])


def build_model_resnet(input_layer, start_neurons, DropoutRatio=0.5, activation=None):
    conv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation=activation, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    pool1 = tf.keras.layers.Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation=activation, padding="same")(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    pool2 = tf.keras.layers.Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation=activation, padding="same")(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    pool3 = tf.keras.layers.Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation=activation, padding="same")(pool3)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8, True)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    pool4 = tf.keras.layers.Dropout(DropoutRatio)(pool4)

    # Middle
    convm = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation=activation, padding="same")(pool4)
    convm = residual_block(convm, start_neurons * 16)
    convm = residual_block(convm, start_neurons * 16, True)

    # 6 -> 12
    deconv4 = tf.keras.layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = tf.keras.layers.concatenate([deconv4, conv4])
    uconv4 = tf.keras.layers.Dropout(DropoutRatio)(uconv4)

    uconv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation=activation, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = residual_block(uconv4, start_neurons * 8, True)

    # 12 -> 25
    # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = tf.keras.layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = tf.keras.layers.concatenate([deconv3, conv3])
    uconv3 = tf.keras.layers.Dropout(DropoutRatio)(uconv3)

    uconv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation=activation, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)

    # 25 -> 50
    deconv2 = tf.keras.layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2])

    uconv2 = tf.keras.layers.Dropout(DropoutRatio)(uconv2)
    uconv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation=activation, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)

    # 50 -> 101
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = tf.keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1])

    uconv1 = tf.keras.layers.Dropout(DropoutRatio)(uconv1)
    uconv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation=activation, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)

    # uconv1 = Dropout(DropoutRatio/2)(uconv1)
    # output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation=activation)(uconv1)
    output_layer = tf.keras.layers.Activation('sigmoid')(output_layer_noActi)

    return output_layer


def build_res_se_modelinput_layer(input_layer, start_neurons, DropoutRatio=0.5, activation=None):
    conv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation=activation, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    conv1 = squeeze_excitation_layer(conv1, start_neurons * 1)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)
    pool1 = tf.keras.layers.Dropout(DropoutRatio / 2)(pool1)

    # 50 -> 25
    conv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation=activation, padding="same")(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    conv2 = squeeze_excitation_layer(conv2, start_neurons * 2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)
    pool2 = tf.keras.layers.Dropout(DropoutRatio)(pool2)

    # 25 -> 12
    conv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation=activation, padding="same")(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    conv3 = squeeze_excitation_layer(conv3, start_neurons * 4)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)
    pool3 = tf.keras.layers.Dropout(DropoutRatio)(pool3)

    # 12 -> 6
    conv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation=activation, padding="same")(pool3)
    conv4 = residual_block(conv4, start_neurons * 8)
    conv4 = residual_block(conv4, start_neurons * 8, True)
    conv4 = squeeze_excitation_layer(conv4, start_neurons * 8)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)
    pool4 = tf.keras.layers.Dropout(DropoutRatio)(pool4)

    # Middle
    convm = tf.keras.layers.Conv2D(start_neurons * 16, (3, 3), activation=activation, padding="same")(pool4)
    convm = residual_block(convm, start_neurons * 16)
    convm = residual_block(convm, start_neurons * 16, True)

    # 6 -> 12
    deconv4 = tf.keras.layers.Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv4 = tf.keras.layers.concatenate([deconv4, conv4])
    uconv4 = tf.keras.layers.Dropout(DropoutRatio)(uconv4)

    uconv4 = tf.keras.layers.Conv2D(start_neurons * 8, (3, 3), activation=activation, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 8)
    uconv4 = residual_block(uconv4, start_neurons * 8, True)
    uconv4 = squeeze_excitation_layer(uconv4, start_neurons * 8)

    # 12 -> 25
    # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3 = tf.keras.layers.Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="valid")(uconv4)
    uconv3 = tf.keras.layers.concatenate([deconv3, conv3])
    uconv3 = tf.keras.layers.Dropout(DropoutRatio)(uconv3)

    uconv3 = tf.keras.layers.Conv2D(start_neurons * 4, (3, 3), activation=activation, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)
    uconv3 = squeeze_excitation_layer(uconv3, start_neurons * 4)

    # 25 -> 50
    deconv2 = tf.keras.layers.Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2])

    uconv2 = tf.keras.layers.Dropout(DropoutRatio)(uconv2)
    uconv2 = tf.keras.layers.Conv2D(start_neurons * 2, (3, 3), activation=activation, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)
    uconv2 = squeeze_excitation_layer(uconv2, start_neurons * 2)

    # 50 -> 101
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    deconv1 = tf.keras.layers.Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="valid")(uconv2)
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1])

    uconv1 = tf.keras.layers.Dropout(DropoutRatio)(uconv1)
    uconv1 = tf.keras.layers.Conv2D(start_neurons * 1, (3, 3), activation=activation, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)
    uconv1 = squeeze_excitation_layer(uconv1, start_neurons * 1)

    # uconv1 = Dropout(DropoutRatio/2)(uconv1)
    # output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(uconv1)
    output_layer_noActi = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation=activation)(uconv1)
    output_layer = tf.keras.layers.Activation('sigmoid')(output_layer_noActi)

    return output_layer


def mnas_skip_block(_input, _filters, depth_mul=1):
    x = tf.keras.layers.DepthwiseConv2D((3, 3), padding='same', depth_multiplier=depth_mul, strides=(1, 1))(_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(_filters, kernel_size=(3, 3), padding="same")(x)
    return tf.keras.layers.BatchNormalization()(x)


def mnas_res_se_block(_input, filters, kernel_size=(3, 3)):
    _inpt = tf.keras.layers.Conv2D(filters, (1, 1), padding="same")(_input)
    x = tf.keras.layers.Conv2D(filters, (1, 1), padding="same")(_inpt)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size, padding='same', depth_multiplier=1, strides=(1, 1))(x)
    x = squeeze_excitation_layer(x, channels=filters)
    x = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Add()([_inpt, x])


def mnas_res_block(_input, filters, kernel_size=(3, 3)):
    _inpt = tf.keras.layers.Conv2D(filters, (1, 1), padding="same", strides=(1, 1))(_input)
    x = tf.keras.layers.Conv2D(filters, (1, 1), padding="same")(_inpt)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size, padding='same', depth_multiplier=1, strides=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, (1, 1), padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Add()([_inpt, x])


def build_mnas_model(_input):
    input_layer = tf.keras.layers.Conv2D(3, (1, 1), padding="same")(_input)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(input_layer)
    pol1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = mnas_skip_block(pol1, _filters=16)
    conv2 = mnas_res_block(conv2, filters=24)
    conv2 = mnas_res_block(conv2, filters=24)

    pol2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

    conv3 = mnas_res_se_block(pol2, 40, kernel_size=(5, 5))
    conv3 = mnas_res_se_block(conv3, 40, kernel_size=(5, 5))
    conv3 = mnas_res_se_block(conv3, 40, kernel_size=(5, 5))

    pol3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)

    conv4 = mnas_res_block(pol3, 80)
    conv4 = mnas_res_block(conv4, 80)
    conv4 = mnas_res_block(conv4, 80)
    conv4 = mnas_res_block(conv4, 80)

    pol4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)

    conv5 = mnas_res_block(pol4, 112)
    conv5 = squeeze_excitation_layer(_input=conv5, channels=112)
    conv5 = mnas_res_block(conv5, 112)
    conv5 = squeeze_excitation_layer(_input=conv5, channels=112)

    pol5 = tf.keras.layers.MaxPooling2D((2, 2))(conv5)

    middle = mnas_res_block(pol5, 160, kernel_size=(5, 5))
    middle = squeeze_excitation_layer(_input=middle, channels=112)
    middle = mnas_res_block(middle, 160, kernel_size=(5, 5))
    middle = squeeze_excitation_layer(_input=middle, channels=112)
    middle = mnas_res_block(middle, 160, kernel_size=(5, 5))
    middle = squeeze_excitation_layer(_input=middle, channels=112)

    middle = mnas_res_block(middle, 320)

    deconv5 = tf.keras.layers.Conv2DTranspose(112, strides=(2, 2), padding="same", kernel_size=(3, 3))(middle)
    uconv5 = tf.keras.layers.concatenate([deconv5, conv5])
    uconv5 = mnas_res_block(uconv5, 112)
    uconv5 = squeeze_excitation_layer(_input=uconv5, channels=112)
    uconv5 = mnas_res_block(uconv5, 112)
    uconv5 = squeeze_excitation_layer(_input=uconv5, channels=112)

    deconv4 = tf.keras.layers.Conv2DTranspose(80, strides=(2, 2), padding="same", kernel_size=(3, 3))(uconv5)
    uconv4 = tf.keras.layers.concatenate([deconv4, conv4])

    uconv4 = mnas_res_block(uconv4, 80)
    uconv4 = mnas_res_block(uconv4, 80)
    uconv4 = mnas_res_block(uconv4, 80)
    uconv4 = mnas_res_block(uconv4, 80)

    deconv3 = tf.keras.layers.Conv2DTranspose(40, strides=(2, 2), padding="same", kernel_size=(3, 3))(uconv4)
    uconv3 = tf.keras.layers.concatenate([deconv3, conv3])

    uconv3 = mnas_res_se_block(uconv3, 40, kernel_size=(5, 5))
    uconv3 = mnas_res_se_block(uconv3, 40, kernel_size=(5, 5))
    uconv3 = mnas_res_se_block(uconv3, 40, kernel_size=(5, 5))

    deconv2 = tf.keras.layers.Conv2DTranspose(24, strides=(2, 2), padding="same", kernel_size=(3, 3))(uconv3)
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2])

    uconv2 = mnas_res_block(uconv2, filters=24)
    uconv2 = mnas_res_block(uconv2, filters=24)

    deconv1 = tf.keras.layers.Conv2DTranspose(16, strides=(2, 2), padding="same", kernel_size=(3, 3))(uconv2)
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1])
    uconv1 = mnas_skip_block(uconv1, _filters=16)
    output_layer = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = tf.keras.layers.Activation('sigmoid')(output_layer)
    return output_layer


def build_ezo_model(_input):
    input_layer = tf.keras.layers.Conv2D(3, (1, 1), padding="same")(_input)
    conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same")(input_layer)
    pol1 = tf.keras.layers.MaxPooling2D((2, 2))(conv1)

    conv2 = mnas_skip_block(pol1, _filters=16)
    conv2 = mnas_res_block(conv2, filters=24)
    conv2 = mnas_res_block(conv2, filters=24)

    pol2 = tf.keras.layers.MaxPooling2D((2, 2))(conv2)

    conv3 = mnas_res_se_block(pol2, 40, kernel_size=(5, 5))
    conv3 = mnas_res_se_block(conv3, 40, kernel_size=(5, 5))
    conv3 = mnas_res_se_block(conv3, 40, kernel_size=(5, 5))

    pol3 = tf.keras.layers.MaxPooling2D((2, 2))(conv3)

    conv4 = mnas_res_block(pol3, 80)
    conv4 = mnas_res_block(conv4, 80)
    conv4 = mnas_res_block(conv4, 80)
    conv4 = mnas_res_block(conv4, 80)

    pol4 = tf.keras.layers.MaxPooling2D((2, 2))(conv4)

    conv5 = mnas_res_block(pol4, 112)
    conv5 = squeeze_excitation_layer(_input=conv5, channels=112)
    conv5 = mnas_res_block(conv5, 112)
    conv5 = squeeze_excitation_layer(_input=conv5, channels=112)

    pol5 = tf.keras.layers.MaxPooling2D((2, 2))(conv5)

    middle = mnas_res_block(pol5, 160, kernel_size=(5, 5))
    middle = squeeze_excitation_layer(_input=middle, channels=112)
    middle = mnas_res_block(middle, 160, kernel_size=(5, 5))
    middle = squeeze_excitation_layer(_input=middle, channels=112)
    middle = mnas_res_block(middle, 160, kernel_size=(5, 5))
    middle = squeeze_excitation_layer(_input=middle, channels=112)

    middle = mnas_res_block(middle, 320)

    factor = tf.keras.layers.Conv2D(1, (1, 1))(middle)
    factor = tf.keras.layers.GlobalAveragePooling2D()(factor)

    deconv5 = tf.keras.layers.Conv2DTranspose(112, strides=(2, 2), padding="same", kernel_size=(3, 3))(middle)
    uconv5 = tf.keras.layers.concatenate([deconv5, conv5])
    uconv5 = mnas_res_block(uconv5, 112)
    uconv5 = squeeze_excitation_layer(_input=uconv5, channels=112)
    uconv5 = mnas_res_block(uconv5, 112)
    uconv5 = squeeze_excitation_layer(_input=uconv5, channels=112)

    deconv4 = tf.keras.layers.Conv2DTranspose(80, strides=(2, 2), padding="same", kernel_size=(3, 3))(uconv5)
    uconv4 = tf.keras.layers.concatenate([deconv4, conv4])

    uconv4 = mnas_res_block(uconv4, 80)
    uconv4 = mnas_res_block(uconv4, 80)
    uconv4 = mnas_res_block(uconv4, 80)
    uconv4 = mnas_res_block(uconv4, 80)

    deconv3 = tf.keras.layers.Conv2DTranspose(40, strides=(2, 2), padding="same", kernel_size=(3, 3))(uconv4)
    uconv3 = tf.keras.layers.concatenate([deconv3, conv3])

    uconv3 = mnas_res_se_block(uconv3, 40, kernel_size=(5, 5))
    uconv3 = mnas_res_se_block(uconv3, 40, kernel_size=(5, 5))
    uconv3 = mnas_res_se_block(uconv3, 40, kernel_size=(5, 5))

    deconv2 = tf.keras.layers.Conv2DTranspose(24, strides=(2, 2), padding="same", kernel_size=(3, 3))(uconv3)
    uconv2 = tf.keras.layers.concatenate([deconv2, conv2])

    uconv2 = mnas_res_block(uconv2, filters=24)
    uconv2 = mnas_res_block(uconv2, filters=24)

    deconv1 = tf.keras.layers.Conv2DTranspose(16, strides=(2, 2), padding="same", kernel_size=(3, 3))(uconv2)
    uconv1 = tf.keras.layers.concatenate([deconv1, conv1])
    uconv1 = mnas_skip_block(uconv1, _filters=16)
    output_layer = tf.keras.layers.Conv2D(1, (1, 1), padding="same", activation=None)(uconv1)
    output_layer = tf.keras.layers.Multiply()([factor, output_layer])
    output_layer = tf.keras.layers.Activation('sigmoid')(output_layer)
    return output_layer



if __name__ == '__main__':
    inpt = tf.keras.layers.Input((224, 224, 1))
    build_ezo_model(inpt)
