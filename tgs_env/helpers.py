import os
import cv2
from util import join_paths
import tensorflow as tf
import numpy as np
import time
# TfRecords
tf.compat.v1.enable_eager_execution()
import tensorflow.contrib.eager as tfe


def get_train_generator(directory, mask_dir, img_dir, read_type=cv2.IMREAD_GRAYSCALE):
    def train_generator():
        _img_dir = join_paths(directory, img_dir)
        _mask_dir = join_paths(directory, mask_dir)

        for imag_file in os.listdir(_img_dir):
            x_img = cv2.imread(join_paths(_img_dir, imag_file), read_type)
            y_img = cv2.imread(join_paths(_mask_dir, imag_file), read_type)
            x_img = np.pad(x_img, ((13, 14), (13, 14)), mode='symmetric').astype(np.float64) / 255.0
            y_img = np.pad(y_img, ((13, 14), (13, 14)), mode='symmetric').astype(np.float64) / 255.0
            img_shape = (x_img.shape[0], x_img.shape[1], 1)
            yield (x_img.reshape(img_shape), y_img.reshape(img_shape))

    return train_generator


def get_test_generator(directory, img_dir, read_type=cv2.IMREAD_GRAYSCALE):
    def test_generator():
        _img_dir = join_paths(directory, img_dir)
        for imag_file in os.listdir(_img_dir):
            x_img = cv2.imread(join_paths(_img_dir, imag_file), read_type)
            yield x_img

    return test_generator


def create_data_set_from_generator(generator, _types, _shapes, buffer_size=100):
    ds = tf.data.Dataset.from_generator(generator, _types, _shapes)
    ds = ds.prefetch(buffer_size)
    return ds


def get_channels(_type):
    if _type == cv2.IMREAD_GRAYSCALE:
        return 1
    elif _type == cv2.IMREAD_COLOR:
        return 3
    return 1


def create_tfrecord(image_mask_list: [()], tfrecord_file, preprocess_callbacks: [callable] = None,
                    _type=cv2.IMREAD_GRAYSCALE):
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for img_file, mask_file in image_mask_list:
            ser_img = serialize_tgs_image(img_file, mask_file, preprocess_callbacks, _type)
            writer.write(ser_img.SerializeToString())


def pad_images(img, mask, pad_shapes=((13, 14), (13, 14))):
    img = np.pad(img, pad_shapes, mode='symmetric').astype(np.float32) / 255.0
    mask = np.pad(mask, pad_shapes, mode='symmetric').astype(np.float32) / 255.0
    return img, mask


def reshape_img(img, mask, shape=(128, 128, 1)):
    img = np.resize(img, shape)
    mask = np.resize(mask, shape)
    return img, mask


def serialize_tgs_image(img_file, mask_file, preprocess_callbacks: [callable], _type):
    img = cv2.imread(img_file, _type)
    mask = cv2.imread(mask_file, _type)
    if preprocess_callbacks:
        for preprocess_callback in preprocess_callbacks:
            img, mask = preprocess_callback(img, mask)

    img = img.astype(np.float32)
    mask = mask.astype(np.float32)

    ser_image = tf.train.Example(features=tf.train.Features(feature={
        'img': tf.train.Feature(float_list=tf.train.FloatList(value=img.reshape(-1).tolist())),
        'mask': tf.train.Feature(float_list=tf.train.FloatList(value=mask.reshape(-1).tolist()))
    }))
    return ser_image


def get_train_val_paths(directory, imgs_dir, masks_dir, percentage=0.8):
    _img_dir = join_paths(directory, imgs_dir)
    _mask_dir = join_paths(directory, masks_dir)
    images = [(join_paths(_img_dir, img_file), join_paths(_mask_dir, img_file)) for img_file in os.listdir(_img_dir)]
    train_size = int(percentage * len(images))
    return images[:train_size], images[train_size:]


def create_deserializer(shape=(128, 128, 1)):
    def deserialize_tgs_image(tfrecord):
        features = {
            'img': tf.FixedLenFeature(shape, tf.float32),
            'mask': tf.FixedLenFeature(shape, tf.float32)
        }
        sample = tf.parse_single_example(tfrecord, features)
        img = sample['img']
        mask = sample['mask']
        return tf.cast(img, tf.float64), tf.cast(mask, tf.float64)
    return deserialize_tgs_image


def create_dataset_from_tfrecord(tf_records, decode_func):
    dataset = tf.data.TFRecordDataset(tf_records)
    dataset = dataset.map(decode_func)
    dataset = dataset.prefetch(64)
    return dataset


if __name__ == '__main__':
    # train_ds = create_dataset_from_tfrecord(['train_images.tfrecord'], create_deserializer())
    # start_time = time.time()
    # for batch in train_ds.take(3200):
    #     cv2.imshow('img', (batch[0].numpy() * 255.0).astype(np.uint8))
    #     cv2.imshow('mask', (batch[1].numpy() * 255.0).astype(np.uint8))
    #     cv2.waitKey()
    # print('time elapsed {}'.format(time.time() - start_time))
    #
    # ds_train = create_data_set_from_generator(get_train_generator('./tgs/train', mask_dir='masks', img_dir='images'),
    #                                           _types=(tf.float64, tf.float64),
    #                                           _shapes=(tf.TensorShape([128, 128, 1]), tf.TensorShape([128, 128, 1])))
    # start_time = time.time()
    # for batch in ds_train.take(3200):
    #     print('.', end='')
    # print('')
    # print('time elapsed {}', format(time.time()-start_time))
    train_files, validation_files = get_train_val_paths('./tgs/train', masks_dir='masks', imgs_dir='images')
    print(len(train_files))
    print(len(validation_files))
    create_tfrecord(train_files, 'train_images.tfrecord', [reshape_img])
    create_tfrecord(validation_files, 'validation_images.tfrecord', [reshape_img])
