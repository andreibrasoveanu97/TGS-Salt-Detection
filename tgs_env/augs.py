import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from albumentations import (RandomContrast, RandomBrightness, ShiftScaleRotate, HorizontalFlip, Compose)
import random


def tf_augs(img, mask):
    seed = random.random()
    res = tf.concat([img, mask], axis=2)
    res = tf.image.random_contrast(res, seed=seed)
    res = tf.image.random_flip_left_right(res, seed=seed)
    res = tf.image.random_flip_up_down(res, seed=seed)
    res = tf.image.random_brightness(res, seed=seed)
    img_aug, mask_aug = tf.split(res, num_or_size_splits=2, axis=2)
    return img_aug, mask_aug


def strong_aug(p=0.9):
    return Compose([
        RandomBrightness(p=0.2, limit=0.2),
        RandomContrast(p=0.1, limit=0.2),
        ShiftScaleRotate(shift_limit=0.1625, scale_limit=0.6, rotate_limit=0, p=0.7),
        HorizontalFlip(p=0.5)
    ], p=p)


def album_aug(img, mask):
    augment = strong_aug()
    data = {'image':img, 'mask':mask}
    augmented = augment(**data)
    return augmented['image'], augmented['mask']


def flip(image_set, mask_set):
    aug = iaa.Fliplr(1)

    imgs, masks = aug.augment_images(image_set, mask_set)
    return imgs, masks


def flip(img, mask, prob=0.5):
    aug = iaa.Fliplr(prob)
    res = np.concatenate([img, mask], axis=2)
    res = aug.augment_image(res)
    return res[:,:,0], res[:,:,1]


def gamma_contrast(img, mask, prob):
    # aug = aat.RandomContrast(p=0.5)
    pass

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
