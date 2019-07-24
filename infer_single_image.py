import random
import time
import os
import cv2
import numpy as np
import torchvision.transforms as transforms
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

image_path = ''
bg_img_dir = '/mnt/ssd/users/semyon/projects/partially_annotated/bg/images'


def get_images_lists(dir_path: str, img_type: str = 'jpg') -> tuple:
    all_files = os.listdir(dir_path)
    images = list(filter(lambda x: img_type in x, all_files))
    amls = list(filter(lambda x: '.aml' in x, all_files))
    all_files = []
    images = [os.path.join(dir_path, im) for im in images]
    amls = [os.path.join(dir_path, im) for im in amls]
    return images, amls


bg_img_names, _ = get_images_lists(bg_img_dir)

opt = TestOptions().parse()

model = create_model(opt)



# prepare A and B crops
# load bg image
img_name = random.choice(bg_img_names)
img = cv2.imread(img_name)
# pick height and width of a crop
height = random.randint(150, img.shape[0] // 2 - 50)
width = random.randint(height // 5, height // 2 - 50)
crop_size = (height + 50) // 2 * 2
# pick a point and add noise
x = random.randint(height, img.shape[1] - width // 2)
y = random.randint(height, img.shape[0] - height // 2)
# make random size crop
crop = img[y-crop_size // 2: y + crop_size // 2, x - crop_size // 2: x + crop_size // 2]
# add noise to the image
noise = np.moveaxis(np.array([np.random.randint(0, 255, (height // 2 * 2, width // 2 * 2))]*3), 0, 2)
img[y - height // 2: y + height // 2, x - width // 2: x + width // 2] = noise
# make noise crop
crop_noise = img[y-crop_size // 2: y + crop_size // 2, x - crop_size // 2: x + crop_size // 2]
# A = crop with noise
# B = original crop
# apply transforms
transform_list = [
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)
# model.set_input({A:..., B:...})
model.set_input({
    'A': transform(cv2.resize(crop, (256, 256))).unsqueeze(0),
    'B': transform(cv2.resize(crop, (256, 256))).unsqueeze(0),
    'bbox': [[0], [10], [0], [10]],
    'A_paths': '',
    'B_paths': ''
})
model.test()
result = model.fake_B
# paste result to bg image
res_np = (np.moveaxis(result.detach().cpu().numpy()[0], 0, 2) + 1) / 2.0 * 255.0
img[y - crop_size // 2: y + crop_size // 2, x - crop_size // 2: x + crop_size // 2] = cv2.resize(res_np, (crop_size, crop_size))
# store bg_image
cv2.imwrite('gan_res.png', img[:, :, ::-1])