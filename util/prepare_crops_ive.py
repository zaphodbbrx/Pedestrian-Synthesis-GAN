import os
import random

import cv2
import numpy as np
import json
from tqdm import tqdm

from aml_utils import load_aml

import ray

def get_images_lists(dir_path: str, img_type: str = 'jpg') -> tuple:
    all_files = os.listdir(dir_path)
    images = list(filter(lambda x: img_type in x, all_files))
    amls = list(filter(lambda x: '.aml' in x, all_files))
    all_files = []
    images = [os.path.join(dir_path, im) for im in images]
    amls = [os.path.join(dir_path, im) for im in amls]
    return images, amls


def close_to_border(img_size: tuple, crop_size: int, x: int, y: int):
    # [left right top bottom]
    height, width = img_size
    res = [0, 0, 0, 0]
    if x - crop_size // 2 < 0:
        res[0] = 1
    if x + crop_size // 2 > width:
        res[1] = 1
    if y - crop_size // 2 < 0:
        res[2] = 1
    if y + crop_size // 2 > height:
        res[3] = 1
    return tuple(res)


def load_image(image_name):
    img = cv2.imread(image_name)
    return img # cv2.resize(img, IMG_SHAPE)

# @ray.remote
def process_img(img_name: str, result_dir: str, subdir: str):
    img = load_image(img_name)
    aml = load_aml(img_name.replace(os.path.splitext(img_name)[1], '.aml'))
    if img is None:
        return
    for obj_idx, obj in enumerate(aml['object']):
        xmin = int(obj['bndbox']['xmin'])
        ymin = int(obj['bndbox']['ymin'])
        xmax = int(obj['bndbox']['xmax'])
        ymax = int(obj['bndbox']['ymax'])
        # if xmin < 50 or ymin < 50 or xmax > int(aml['size']['width']) - 50 or ymax > int(aml['size']['height']) - 50:
        #     continue

        height = ymax - ymin
        width = xmax - xmin
        x = (xmax + xmin) // 2
        y = (ymax + ymin) // 2
        if not obj['name'] == 'eOC_Human' or height < 140 or width < 50 or width > height:
            continue
        # make crop
        crop_size = height + 50
        # skip close to corner objects
        xmin_cropped = max(x - crop_size // 2, 0)
        xmax_cropped = min(x + crop_size // 2, img.shape[1])
        ymin_cropped = max(y - crop_size // 2, 0)
        ymax_cropped = min(y + crop_size // 2, img.shape[0])

        x_crop = x - xmin_cropped
        y_crop = y - ymin_cropped
        crop = img[ymin_cropped:ymax_cropped, xmin_cropped:xmax_cropped, :]
        # resize crop to 256x256
        crop = cv2.resize(crop, (256, 256))
        x_crop = int(x_crop * 256 / crop_size)# - width // 2
        y_crop = int(y_crop * 256 / crop_size)# - height // 2
        width = int(width * 256 / crop_size)
        height = int(height * 256 / crop_size)
        if width < 50 or height < 140:
            continue
        # copy crop
        crop_noise = crop.copy()
        # paste noise in bbox
        noise = np.moveaxis(np.array([np.random.randint(0, 255, (256, 256))]*3), 0, 2)
        noise_mask = np.zeros((256, 256, 3)).astype(np.bool)
        noise_mask[y_crop - height // 2: y_crop + height // 2, x_crop - width // 2: x_crop + width // 2, :] = True
        crop_noise[noise_mask] = noise[noise_mask]
        # concat orig and noise
        crop_res = np.hstack([crop, crop_noise])
        # create annotation_dict
        annotation_dict = {
            "y": y_crop - height // 2 + 1,
            "x": x_crop - width // 2 + 1,
            # "w": width,
            # "h": height
            "w": x_crop + width // 2,
            "h": y_crop + height // 2,
        }
        # save concatenated img
        fname = os.path.splitext(os.path.basename(img_name))[0]
        img_fname = f'{obj_idx}_{fname}.png'
        json_fname = f'{obj_idx}_{fname}.json'
        cv2.imwrite(os.path.join(result_dir, 'images', subdir, img_fname), crop_res)
        # save annotation_dict to json
        with open(os.path.join(result_dir, 'bbox', subdir, json_fname), 'w') as fp:
            json.dump(annotation_dict, fp)


images_dir = '/mnt/ssd/users/semyon/projects/partially_annotated/fully_annotated/iv_cloud_20181210/images'
result_dir = '/mnt/ssd/users/semyon/projects/partially_annotated/partially_annotated_2_synt_gan'
imgs, amls = get_images_lists(images_dir)

# random.shuffle(imgs)

train_imgs = sorted(imgs)[:int(0.9 * len(imgs))]
valid_imgs = sorted(imgs)[int(0.9 * len(imgs)):]

# ray.init(num_cpus=8, ignore_reinit_error=True, include_webui=False)
for img_name in tqdm(train_imgs):
    process_img(img_name, result_dir, 'train')

for img_name in tqdm(valid_imgs):
    process_img(img_name, result_dir, 'valid')