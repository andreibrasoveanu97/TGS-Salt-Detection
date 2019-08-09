import numpy as np
import cv2
import pandas as pd
import os


def encode_rle(imag: np.array, order='F', object_value=255):
    flatten_img = imag.reshape(-1, order=order)
    runs = []
    length = 0
    last_pixel = 0
    for index, pixel in enumerate(flatten_img):
        if pixel == object_value:
            if last_pixel != pixel:
                runs.append(index + 1)
                length = 1

            else:
                length += 1
        else:
            if last_pixel == object_value:
                runs.append(length)
                length = 0

        last_pixel = pixel
    return runs


def decode_rle(rle_code: [], rows, cols, dtype=np.uint8, object_value=255):
    imag = np.zeros(rows * cols, dtype=dtype)
    for index in range(0, len(rle_code), 2):
        pos_in_img = rle_code[index] - 1
        length = rle_code[index + 1]
        imag[pos_in_img: pos_in_img + length] = object_value

    return imag.reshape((rows, cols))


def join_paths(dir_name, file_name, sep='/'):
    return dir_name + sep + file_name


def get_paths(directory, mask_dir, img_dir):
    img_dir = join_paths(directory, img_dir)
    mask_dir = join_paths(directory, mask_dir)
    ids = []
    mask_files = []
    img_files = []
    for img_file in os.listdir(img_dir):
        #     print(join_paths(mask_dir, img_file))
        ids.append(img_file.split('.')[0])
        mask_files.append(join_paths(mask_dir, img_file))
        img_files.append(join_paths(img_dir, img_file))

    df = pd.DataFrame()
    df['id'] = ids
    df['mask'] = mask_files
    df['img'] = img_files
    df.set_index('id')
    return df


def create_coverage_stratas(imags_df, _type=cv2.IMREAD_GRAYSCALE):
    coverages = []
    cover_stratas = []
    for index, row in imags_df.iterrows():
        mask = cv2.imread(row['mask'], _type) // 255
        coverage = np.sum(mask) / float(mask.size)
        coverages.append(coverage)
        cover_stratas.append(int(coverage * 10))
    imags_df['coverage'] = coverages
    imags_df['coverage_strata'] = cover_stratas
    return imags_df


#
# data = get_paths('./drive/My Drive/tgs/train', 'masks', 'images')
# data = data.set_index('id')

if __name__ == '__main__':
    image = cv2.imread('./tgs/train/masks/0a0814464f.png', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('der true mask', np.transpose(image))
    rns = encode_rle(image)
    img = decode_rle(rns, rows=101, cols=101)
    cv2.imshow('der gen mask', img)
    print(img == image)

    print(rns)
    cv2.waitKey()
