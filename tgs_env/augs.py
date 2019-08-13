import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
from albumentations import (HorizontalFlip, RandomBrightness, RandomContrast, ShiftScaleRotate)
import tfAugmentor as tfa


def tf_augs(img, mask, label = 'segmentation_mask'):
    list = {'img': img,
            'mask': mask}
    a = tfa.Augmentor(list, label=[label])
    a.flip_left_right(probability=0.5)
    augmented = a.out
    return augmented['img'], augmented['mask']


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
