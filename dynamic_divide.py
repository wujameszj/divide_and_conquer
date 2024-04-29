import os
import glob
import math
import pickle
import shutil
import argparse
from time import time

from tqdm import tqdm
import cv2
from cv2 import imread, imwrite
import numpy as np
import scipy.ndimage
from PIL import Image

from torchvision.transforms import functional as F
import torch
import torch.nn.functional as torch_F
import torch.nn as nn



parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-i', '--input')
parser.add_argument('-m', '--mask')
parser.add_argument('-o', '--output')
parser.add_argument('--model', type=str, default='~/ProPainter/weights/ProPainter.pth')
parser.add_argument('-r', "--raft_iter", type=int, default=10, help='Iterations for RAFT inference.')
parser.add_argument('-s', "--resize_ratio", type=float, default=.5, help='Resize scale for processing video.')
parser.add_argument('-t', "--ref_stride", type=int, default=30, help='Stride of global reference frames.')
parser.add_argument('-n', "--neighbor_length", type=int, default=30, help='Length of local neighboring frames.')
parser.add_argument('-v', "--subvideo_length", type=int, default=200, help='Length of sub-video for long video inference.')
parser.add_argument('-h', '--height', type=int, default=-1, help='Height of the processing video.')
parser.add_argument('-w', '--width', type=int, default=-1, help='Width of the processing video.')
args = parser.parse_args()


def invert(path, save_directory):
    inverted_image = 255 - imread(path, cv2.IMREAD_GRAYSCALE)
    save_path = os.path.join(save_directory, os.path.basename(path))
    imwrite(save_path, inverted_image)

def binary_mask(mask, th=0.1):
    mask[mask > th] = 1
    mask[mask <= th] = 0
    return mask

def resize_images(inverted_mask_dir, resized_dir, size):
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)

    for filename in tqdm(os.listdir(inverted_mask_dir), desc='resizing', leave=False):
        img = Image.open(os.path.join(inverted_mask_dir, filename))
        img = img.resize(size, Image.LANCZOS)
        img.save(os.path.join(resized_dir, filename))


if args.width == 728 and args.height == 408:
    min_subvideo_length, max_subvideo_length = 40, 100
else:
    min_subvideo_length = 200 if args.resize_ratio == .5 else 130
    max_subvideo_length = 500 if args.resize_ratio == .5 else 300


# reference : read_mask func(inference_propainter.py)
def dilate_mask(mask_inverted_dir, dilated_mask_dir, mask_dilates=5):
    os.makedirs(dilated_mask_dir, exist_ok=True)
    mask_files = os.listdir(mask_inverted_dir)

    for mask_file in mask_files:
        mask_img = Image.open(os.path.join(mask_inverted_dir, mask_file)) #.convert('L')
        mask_np = np.array(mask_img)
        if mask_dilates > 0:
            mask_dilated = scipy.ndimage.binary_dilation(mask_np, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_dilated = binary_mask(mask_np).astype(np.uint8)

        mask_dilated_img = Image.fromarray(mask_dilated * 255)
        mask_dilated_img.save(os.path.join(dilated_mask_dir, mask_file))


input_color_dir = args.input
input_mask_dir = args.mask
input_mask_dir_inverted = os.path.join(args.output, "inverted_masks")
resized_mask_dir = os.path.join(args.output, "resized_masks")
mask_invert_dilate_dir = os.path.join(args.output, "inverted_dilated_masks")
downsampled_mask_dir = os.path.join(args.output, "downampled_masks")

output_dir = os.path.join(args.output, "result")

if not os.path.exists(input_mask_dir_inverted) or len(os.listdir(input_mask_dir_inverted)) != len(os.listdir(input_mask_dir)):
    os.makedirs(input_mask_dir_inverted, exist_ok=True)
    for mask_name in tqdm(os.listdir(input_mask_dir), desc='inverting', leave=False):
        path = os.path.join(input_mask_dir, mask_name)
        invert(path, input_mask_dir_inverted)

resize_images(input_mask_dir_inverted, resized_mask_dir, (int(1280*args.resize_ratio), int(720*args.resize_ratio)))

mask_dilates = 4 # default:4 (Can be changed with the --mask_dilation option)
dilate_mask(resized_mask_dir, mask_invert_dilate_dir, mask_dilates)

if not os.path.exists(downsampled_mask_dir):
    os.makedirs(downsampled_mask_dir)

for dilated_image_path in glob.glob(os.path.join(mask_invert_dilate_dir, '*.png')):
    # reference : propainter.py
    dilated_image = Image.open(dilated_image_path)
    dilated_image_tensor = F.to_tensor(dilated_image).unsqueeze(0)
    ds_mask_in = torch_F.interpolate(dilated_image_tensor, scale_factor=1/4, mode='nearest')
    mask_pool = nn.MaxPool2d(kernel_size=(7, 7), stride=(3, 3), padding=(3, 3))(ds_mask_in)

    # reference : sparse_transformer.py
    h, w = mask_pool.shape[2], mask_pool.shape[3]
    new_h = math.ceil(h / 5) * 5
    new_w = math.ceil(w / 9) * 9
    padded_mask = F.pad(mask_pool, (0, new_w - w, new_h - h, 0))
    final_ds_mask = nn.MaxPool2d(kernel_size=(5, 9), stride=(5, 9), padding=(0, 0))(padded_mask)

    final_ds_mask_image = F.to_pil_image(final_ds_mask.squeeze(0))
    base_name = os.path.basename(dilated_image_path)
    final_ds_mask_image.save(os.path.join(downsampled_mask_dir, f'{base_name}'))

ds_masks = sorted(os.listdir(downsampled_mask_dir))
ds_masks_len = len(ds_masks)

th_list = []
for i in range(max_subvideo_length+1):
    if args.width == 728 and args.height == 408:
        th_list.append(int( (12.0*1024-12.99246*i-7510.35) / (0.3754*i+182.76) )) # resize 0.5, neighbor length 40, ref stride 20, subvideo length 100, 95% quantile regression
    elif args.resize_ratio == .5:
        th_list.append(int((14*1024-0.16481*i-6523.19) / (0.8749*i+51.14))) # resize 0.5, 95% quantile regression
    elif args.resize_ratio == .6:
        if i <= 200:
            th_list.append(int((14*1024-31.45013*i-2663.58) / (0.14078*i+196.60))) # resize 0.6
        else:
            th_list.append(int((14*1024-31.45013*200-2663.58) / (0.14078*200-196.60)))

divide_list = [] # i.e. [[0, 19], [20, 39], [40, 60], ...]

start_idx = 0
use_frame = min_subvideo_length # minimum subvideo length
while start_idx < ds_masks_len:
    end_idx = min(ds_masks_len, start_idx + use_frame)
    mask_area_results = []
    for i in range(start_idx, end_idx, args.neighbor_length//2):
        pixel_sum = None
        for j in range(max(i - args.neighbor_length//2, start_idx), min(i + args.neighbor_length//2, end_idx)):
            ds_mask_name = ds_masks[j]
            ds_mask_image = Image.open(os.path.join(downsampled_mask_dir, ds_mask_name))
            ds_mask_image_np = np.array(ds_mask_image)

            pixel_sum = ds_mask_image_np if pixel_sum is None else pixel_sum + ds_mask_image_np

        non_zero_pixels = np.count_nonzero(pixel_sum)
        mask_area_results.append(non_zero_pixels)

    if max(mask_area_results) < th_list[use_frame] and use_frame < max_subvideo_length:
        use_frame = use_frame + 1
    else:
        use_frame = use_frame - 1
        end_idx = min(ds_masks_len, start_idx + use_frame)
        divide_list.append([start_idx, end_idx-1])
        start_idx = end_idx
        use_frame = min_subvideo_length


old_start_idx = divide_list[-1][0]
if ds_masks_len - old_start_idx < 50:
    divide_list[-1][0] = ds_masks_len - 50
skip_frame = old_start_idx - divide_list[-1][0]


for i, subvideo in enumerate(divide_list):
    output_subdir = os.path.join(output_dir, f"temp{i}")
    os.makedirs(output_subdir, exist_ok=True)

    start_idx = subvideo[0]
    end_idx = subvideo[1]
    input_color_images = sorted(os.listdir(input_color_dir))[start_idx:end_idx+1] 
    input_mask_images_inverted = sorted(os.listdir(input_mask_dir_inverted))[start_idx:end_idx+1]
    
    temp_color_dir = f"{output_dir}/temp_color"
    temp_mask_dir = f"{output_dir}/temp_mask"

    if os.path.exists(temp_color_dir):
        shutil.rmtree(temp_color_dir)
    if os.path.exists(temp_mask_dir):
        shutil.rmtree(temp_mask_dir)
    os.makedirs(temp_color_dir)
    os.makedirs(temp_mask_dir)

    for color_image, mask_image in zip(input_color_images, input_mask_images_inverted):
        shutil.copy(os.path.join(input_color_dir, color_image), temp_color_dir)
        shutil.copy(os.path.join(input_mask_dir_inverted, mask_image), temp_mask_dir)

#    cmd = f'python infer.py -r {args.raft_iter} -s {args.resize_ratio} -n {args.neighbor_length} -v {args.subvideo_length} -t {args.ref_stride} --model {args.model} -i {temp_color_dir} -m {temp_mask_dir} -o {output_subdir} --save_frames' \
    cmd = f'python infer.py -r {args.raft_iter} -h {args.height} -w {args.width} -n {args.neighbor_length} -v {args.subvideo_length} -t {args.ref_stride} --model {args.model} -i {temp_color_dir} -m {temp_mask_dir} -o {output_subdir} --save_frames' \
        + (f' --skip_frame {skip_frame}' if subvideo==divide_list[-1] else '') + (' --print_args' if i==0 or subvideo==divide_list[-1] else '')
    os.system(cmd)

    shutil.rmtree(temp_color_dir)
    shutil.rmtree(temp_mask_dir)


dir_quantity = len(divide_list)
final_result_dir = os.path.join(output_dir, f"final_result_dynamic")
if not os.path.exists(final_result_dir):
    os.makedirs(final_result_dir)

for i in range(dir_quantity):
    last_dir_flag = False
    temp_color_dir = os.path.join(output_dir, f"temp{i}")
    temp_color_files = sorted(os.listdir(temp_color_dir))

    if i != dir_quantity-1:
        os.system(f'mv {temp_color_dir}/* {final_result_dir}/')
        continue
    
    for num, temp_color_file in enumerate(temp_color_files, start=skip_frame):
        old_path = os.path.join(temp_color_dir, temp_color_file)
        os.system(f'mv {old_path} {final_result_dir}/')
    shutil.rmtree(temp_color_dir)
