import os
import cv2
from util import join_paths
import tensorflow as tf

tf.enable_eager_execution()


def get_train_generator(directory, mask_dir, img_dir, read_type=cv2.IMREAD_GRAYSCALE):
    def train_generator():
        _img_dir = join_paths(directory, img_dir)
        _mask_dir = join_paths(directory, mask_dir)

        for imag_file in os.listdir(_img_dir):
            x_img = cv2.imread(join_paths(_img_dir, imag_file), read_type)
            y_img = cv2.imread(join_paths(_mask_dir, imag_file), read_type)
            yield (x_img, y_img)

    return train_generator


def get_test_generator(directory, img_dir, read_type=cv2.IMREAD_GRAYSCALE):
    def test_generator():
        _img_dir = join_paths(directory, img_dir)
        for imag_file in os.listdir(_img_dir):
            x_img = cv2.imread(join_paths(_img_dir, imag_file), read_type)
            yield x_img

    return test_generator


def create_data_set_from_generator(generator, _types, _shapes):
    return tf.data.Dataset.from_generator(generator, _types, _shapes)


ds_train = create_data_set_from_generator(get_train_generator('./tgs/train', mask_dir='masks', img_dir='images'),
                                          _types=(tf.uint8, tf.uint8),
                                          _shapes=(tf.TensorShape([101, 101]), tf.TensorShape([101, 101])))

ds_test = create_data_set_from_generator(get_test_generator('./tgs/test', img_dir='images'),
                                         _types=tf.uint8,
                                         _shapes=tf.TensorShape([101, 101]))

if __name__ == '__main__':
    for imag in ds_test.take(1):
        cv2.imshow('damn', imag.numpy())
        cv2.waitKey()
