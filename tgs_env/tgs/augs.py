import imgaug as ia
import imgaug.augmenters as iaa
from helpers import reshape_img
def flip(image_set, mask_set):
    aug = iaa.Fliplr(1)

    imgs, masks = aug.augment_images(image_set, mask_set)
    return imgs, masks

def rotate(image_set, mask_set, degrees = 90):
    aug = iaa.Affine(rotate=degrees)

    imgs = aug.augment_images(image_set)
    masks = aug.augment_images(mask_set)
    return imgs, masks

def scale(image_set, mask_set):
    aug = iaa.Affine(scale=1.2674)

    imgs = aug.augment_images(image_set)
    masks = aug.augment_images(mask_set)
    return imgs, masks