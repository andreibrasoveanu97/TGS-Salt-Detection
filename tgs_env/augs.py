import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from albumentations import (ShiftScaleRotate, HorizontalFlip, Compose)
import random


def tf_augs(img, mask):
    seed = random.random()
    res = tf.concat([img, mask], axis=2)
    res = tf.image.random_contrast(res, seed=seed, lower=0.1, upper=0.8)
    res = tf.image.random_flip_left_right(res, seed=seed)
    res = tf.image.random_flip_up_down(res, seed=seed)
    res = tf.image.random_brightness(res, seed=seed, max_delta=0.9)
    img_aug, mask_aug = tf.split(res, num_or_size_splits=2, axis=2)
    return img_aug, mask_aug


def get_tf_pad(padding=((13, 14), (13, 14))):
    def tf_pad(img, mask):
        return tf.pad(img, padding, 'SYMMETRIC'), tf.pad(mask, padding, 'SYMMETRIC')
    return tf_pad


def norm_and_float(img, mask):
    return tf.cast(img, tf.float64) / 255.0, tf.cast(mask, tf.float64) / 255.0


def tf_alldir_augs(img, mask):
    seed = random.random()
    res = tf.concat([img, mask], axis=2)
    res = tf.image.random_flip_left_right(res, seed=seed)
    res = tf.image.random_flip_up_down(res, seed=seed)
    img_aug, mask_aug = tf.split(res, num_or_size_splits=2, axis=2)
    return img_aug, mask_aug


def tf_alldir_rotate_augs(img, mask):
    seed = random.random()
    res = tf.concat([img, mask], axis=2)
    res = tf.image.random_flip_left_right(res, seed=seed)
    res = tf.image.random_flip_up_down(res, seed=int(seed))
    res = tf.image.rot90(res, seed % 3)
    img_aug, mask_aug = tf.split(res, num_or_size_splits=2, axis=2)
    return img_aug, mask_aug


def expand_dims(img, mask):
    return tf.expand_dims(img, -1), tf.expand_dims(mask, -1)


def strong_aug(p=0.9):
    return Compose([
        ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7),
        HorizontalFlip(p=0.5)
    ], p=p)


def album_aug(img, mask):
    augment = strong_aug()
    data = {'image': img, 'mask': mask}
    augmented = augment(**data)
    return augmented['image'], augmented['mask']


def flip(image_set, mask_set):
    aug = iaa.Fliplr(1)

    imgs, masks = aug.augment_images(image_set, mask_set)
    return imgs, masks


def get_rotate(degrees=90):

    def rotate(image_set, mask_set):
        aug = iaa.Affine(rotate=degrees)
        imgs = aug.augment_images(image_set)
        masks = aug.augment_images(mask_set)
        return imgs, masks
    return rotate


def get_rotate_both(degrees=90):

    def rotate_both(image_set, mask_set):
        imgs1, masks1 = get_rotate(degrees)(image_set, mask_set)
        imgs2, masks2 = get_rotate(-degrees)(image_set, mask_set)
        imgs1 = np.append(imgs1, imgs2)
        masks1 = np.append(masks1, masks2)
        return imgs1, masks1
    return rotate_both


def get_resize(width=128, height=128):

    def resize(image_set, mask_set):
        aug = iaa.Resize({"height": height, "width": width})
        imgs = aug.augment_images(image_set)
        masks = aug.augment_images(mask_set)
        return imgs, masks

    return resize
