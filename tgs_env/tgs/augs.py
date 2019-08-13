import imgaug.augmenters as iaa
import numpy as np


def flip(image_set, mask_set):
    aug = iaa.Fliplr(1)

    imgs, masks = aug.augment_images(image_set, mask_set)
    return imgs, masks

def flip(img, mask, prob=0.5):
    aug = iaa.Fliplr(prob)
    res = np.concatenate([img, mask], axis=2)
    res = aug.augment_image(res)
    return res[:,:,0], res[:,:,1]

def gamma_contrast(img, mask, prob)

def get_rotate(degrees=90):

    def rotate(image_set, mask_set):
        aug = iaa.Affine(rotate=degrees)
        imgs = aug.augment_images(image_set)
        masks = aug.augment_images(mask_set)
        return imgs, masks
    return rotate


# def get_rotate_both(degrees=90):
#
#     def rotate_both(image_set, mask_set):
#         imgs1, masks1 = get_rotate(degrees)(image_set, mask_set)
#         imgs2, masks2 = get_rotate(-degrees)(image_set, mask_set)
#         image_set.extend(imgs1)
#         image_set.extend(imgs2)
#         mask_set.extend(masks1)
#         mask_set.extend(masks2)
#         return image_set, mask_set
#     return rotate_both

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
