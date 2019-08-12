import os
import cv2
from util import join_paths
import tensorflow as tf
import numpy as np
import time

# TfRecords
# tf.compat.v1.enable_eager_execution()
# import tensorflow.contrib.eager as tfe


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


def create_data_set_from_generator(generator, _types, _shapes, buffer_size=100, batch_size=20):
    ds = tf.data.Dataset.from_generator(generator, _types, _shapes)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size)
    return ds


def get_channels(_type):
    if _type == cv2.IMREAD_GRAYSCALE:
        return 1
    elif _type == cv2.IMREAD_COLOR:
        return 3
    return 1


def create_tfrecords(image_mask_list: [()], tf_directory, preprocess_callbacks: [callable] = None,
                     _type=cv2.IMREAD_GRAYSCALE):
    for img_file, mask_file in image_mask_list:
        imgs = [cv2.imread(img_file, _type)]
        masks = [cv2.imread(mask_file, _type)]
        file_to_save = tf_directory + '/' + os.path.basename(img_file).split('.')[0]
        if preprocess_callbacks:
            for preprocess_callback in preprocess_callbacks:
                imgs, masks = preprocess_callback(imgs, masks)
        for index, (img, mask) in enumerate(zip(imgs, masks)):
            # cv2.imshow('img', img)
            # cv2.imshow('maks', mask)
            # cv2.waitKey()
            with tf.io.TFRecordWriter(file_to_save + str(index) + '.tfrecord') as writer:
                img = img.astype(np.float32)
                mask = mask.astype(np.float32)
                # cv2.imshow('img', (img * 255.0).astype(np.uint8))
                # cv2.imshow('maks', (mask * 255.0).astype(np.uint8))
                # cv2.waitKey()
                ser_image = tf.train.Example(features=tf.train.Features(feature={
                    'img': tf.train.Feature(float_list=tf.train.FloatList(value=img.reshape(-1).tolist())),
                    'mask': tf.train.Feature(float_list=tf.train.FloatList(value=mask.reshape(-1).tolist()))
                    # 'file': tf.train.Feature(bytes_list=tf.train.BytesList(value=[file_to_save.encode('utf-8')]))
                }))

                writer.write(ser_image.SerializeToString())


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


def get_reshaper(shape=(128, 128)):
    def reshape_imgs(imgs, masks):
        _imgs = []
        _masks = []
        for img, mask in zip(imgs, masks):
            # cv2.imshow('img', img)
            # cv2.imshow('mask', mask)
            # cv2.waitKey()
            (_img, _mask) = reshape_img(img, mask, shape=shape)
            _imgs.append(_img)
            _masks.append(_mask)
            # cv2.imshow('_img', _img)
            # cv2.imshow('_mask', _mask)
            # cv2.waitKey()
        return _imgs, _masks
    return reshape_imgs


def reshape_img(img, mask, shape=(128, 128)):
    _shape = (shape[0], shape[1], 1)
    img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC).reshape(_shape)
    mask = cv2.resize(mask, shape, interpolation=cv2.INTER_CUBIC).reshape(_shape)
    # cv2.imshow('r_img', img)
    # cv2.imshow('r_mask', mask)
    # cv2.waitKey()
    return img, mask


def serialize_tgs_image(img_file, mask_file, preprocess_callbacks: [callable], _type):
    img = cv2.imread(img_file, _type)
    mask = cv2.imread(mask_file, _type)
    if preprocess_callbacks:
        for preprocess_callback in preprocess_callbacks:
            img, mask = preprocess_callback(img, mask)

    img = img.astype(np.float32)
    mask = mask.astype(np.float32)
    cv2.imshow('img', img)
    cv2.waitKey()
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
            'img': tf.io.FixedLenFeature(shape, tf.float32),
            'mask': tf.io.FixedLenFeature(shape, tf.float32)
            # 'file': tf.io.FixedLenFeature([], tf.string)
        }
        sample = tf.io.parse_single_example(tfrecord, features)
        _img = sample['img']
        mask = sample['mask']
        return tf.cast(_img, tf.float64), tf.cast(mask, tf.float64)

    return deserialize_tgs_image


def create_dataset_from_directory(directory, decode_func, batch_size=20, buffer_size=40, parallel_readers=10,
                                  parallel_calls=10):
    files = tf.data.Dataset.list_files(directory + "/*.tfrecord")
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=parallel_readers)
    # dataset = files.apply(tf.contrib.data.parallel_interleave(
    #     tf.data.TFRecordDataset, cycle_length=parallel_readers))
    # dataset = dataset.map(map_func=decode_func)
    # dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.map(map_func=decode_func, num_parallel_calls=parallel_calls)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size)

    return dataset


def create_dataset_from_tfrecord(tf_records, decode_func, batch_size):
    dataset = tf.data.TFRecordDataset(tf_records)
    dataset = dataset.map(decode_func)
    dataset = dataset.map(batch_size)
    dataset = dataset.prefetch(40)
    return dataset


if __name__ == '__main__':
    train_ds = create_dataset_from_directory('./train_records', create_deserializer())

    # train_ds = create_dataset_from_tfrecord(['train_images.tfrecord'], create_deserializer())
    start_time = time.time()
    for batch in train_ds.take(160):
        img2 = batch[0][0]

        cv2.imshow('img', (img2.numpy()).astype(np.uint8))
        print(batch[2][0].numpy().decode('utf-8'))
        # img2 = cv2.imread(batch[2][0].numpy().decode('utf-8') + ', cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('org', img2)
        print(img2.shape)
        cv2.waitKey()
    print('')
    # print('time elapsed {}'.format(time.time() - start_time))
    # #
    # ds_train = create_data_set_from_generator(get_train_generator('./tgs/train', mask_dir='masks', img_dir='images'),
    #                                           _types=(tf.float64, tf.float64),
    #                                           _shapes=(tf.TensorShape([128, 128, 1]), tf.TensorShape([128, 128, 1])))
    #
    # start_time = time.time()
    # for batch in ds_train.take(160):
    #     print('.', end='')
    # print('')
    # print('time elapsed {}', format(time.time()-start_time))
    # train_files, validation_files = get_train_val_paths('./tgs/train', masks_dir='masks', imgs_dir='images')
    # print(len(train_files))
    # print(len(validation_files))
    # create_tfrecords(train_files, './train_records', [get_reshaper()])
    # create_tfrecord(validation_files, 'validation_images.tfrecord', [reshape_img])
