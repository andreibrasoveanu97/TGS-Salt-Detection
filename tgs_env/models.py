from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import cv2
# import tensorflow.contrib.eager as tfe
import numpy as np
from layers import UnetDecodeLayer, UnetMiddleLayer, UnetEncodeLayer
tf.compat.v1.enable_eager_execution()


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

    @tf.contrib.eager.defun
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


if __name__ == '__main__':
    model = UnetModel()
    img = cv2.imread('./tgs/train/images/0a7e067255.png', cv2.IMREAD_GRAYSCALE)
    out_img = model(np.reshape(cv2.resize(img, (128, 128)), (1, 128, 128, 1)).astype(np.float64))
    out_img = np.reshape(out_img, (128, 128))
    cv2.imshow('mask', out_img)
    cv2.imshow('img', cv2.resize(img, (128, 128)))

    cv2.waitKey()
