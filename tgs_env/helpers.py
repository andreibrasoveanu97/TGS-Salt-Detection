import os
import cv2
from util import join_paths
import tensorflow as tf
tf.enable_eager_execution()


def train_generator(directory, mask_dir, img_dir):
    img_dir = join_paths(directory, img_dir)
    mask_dir = join_paths(directory, mask_dir)

    for imag_file in os.listdir(img_dir):
        x_img = cv2.imread(join_paths(img_dir, imag_file), cv2.IMREAD_GRAYSCALE)
        y_img = cv2.imread(join_paths(mask_dir, imag_file), cv2.IMREAD_GRAYSCALE)
        yield (x_img, y_img)


def create_data_set_from_generator(generator):
    return tf.data.Dataset.from_generator(generator, ())