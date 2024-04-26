# -*- coding: utf-8 -*-
import os
from time import time 

import cv2
import argparse
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import trange, tqdm

import torch
from torch import empty
from torch.cuda import max_memory_allocated, max_memory_reserved, reset_peak_memory_stats, empty_cache
import torchvision

from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter_inference import InpaintGenerator
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device

import warnings
warnings.filterwarnings("ignore")

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'


def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def resize_frames(frames, size=None, algorithm=Image.LANCZOS):
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        frames = [f.resize(process_size, algorithm) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size, algorithm) for f in frames]

    return frames, process_size, out_size


def read_frame_from_videos(frame_root):
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec') # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info['video_fps']
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in tqdm(fr_lst, leave=False, desc='reading frames'):
            frames.append(Image.open(os.path.join(frame_root, fr)))
        fps = None
    size = frames[0].size

    return frames, fps, size, video_name


def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask
  
  
# read frame-wise masks
def read_mask(mpath, length, size, flow_mask_dilates=8, mask_dilates=5, algorithm=Image.NEAREST):  # Image.LANCZOS
    masks_img = []
    masks_dilated = []
    flow_masks = []
    
    if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
       masks_img = [Image.open(mpath)]
    else:  
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))
          
    for mask_img in tqdm(masks_img, leave=False, desc='reading masks'):
        if size is not None:
            mask_img = mask_img.resize(size, algorithm)
        mask_img = np.array(mask_img.convert('L'))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))
    
    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated



def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index



if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = get_device()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--video', type=str, default='inputs/object_removal/bmx-trees', help='Path of the input video or image folder.')
    parser.add_argument(
        '-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask', help='Path of the mask(s) or mask folder.')
    parser.add_argument(
        '-o', '--output', type=str, default='results', help='Output folder. Default: results')
    parser.add_argument(
        '-p', "--resize_ratio", type=float, default=1.0, help='Resize scale for processing video.')
    parser.add_argument(
        '--height', type=int, default=-1, help='Height of the processing video.')
    parser.add_argument(
        '--width', type=int, default=-1, help='Width of the processing video.')
    parser.add_argument(
        '--mask_dilation', type=int, default=4, help='Mask dilation for video and flow masking.')
    parser.add_argument(
        '-t', "--ref_stride", type=int, default=40, help='Stride of global reference frames.')
    parser.add_argument(
        '-n', "--neighbor_length", type=int, default=40, help='Length of local neighboring frames.')
    parser.add_argument(
        "--subvideo_length", type=int, default=999, help='Length of sub-video for long video inference.')
    parser.add_argument(
        '-r',  "--raft_iter", type=int, default=20, help='Iterations for RAFT inference.')
    parser.add_argument(
        '--save_frames', action='store_true', help='Save output frames. Default: False')
    parser.add_argument(
        '--model', type=str, default='weights/ProPainter.pth', help='Path of propainter model')
    parser.add_argument('--record_time', action='store_true')
    parser.add_argument('--vram_stat', action='store_true')
    parser.add_argument('--keep_pbar', action='store_true')
    parser.add_argument("--skip_frame",type=int, default=0)
    args = parser.parse_args()

    if args.record_time: st = time()

    frames, fps, size, video_name = read_frame_from_videos(args.video)
    ori_reso_frames = frames

    if not args.width == -1 and not args.height == -1:
        size = (args.width, args.height)
    if not args.resize_ratio == 1.0:
        size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))

    frames, size, out_size = resize_frames(frames, size)

    save_root = args.output
    if not os.path.exists(save_root):
        os.makedirs(save_root, exist_ok=True)

    frames_len = len(frames)
    flow_masks, masks_dilated = read_mask(args.mask, frames_len, size, 
                                            flow_mask_dilates=args.mask_dilation,
                                            mask_dilates=args.mask_dilation)
    w, h = size


    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1    
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)


    ##############################################
    # set up RAFT and flow completion model
    ##############################################
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)

    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
                                    model_dir='weights', progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()
    fix_flow_complete = fix_flow_complete.half()


    ##############################################
    # set up ProPainter model
    ##############################################
    if args.model == 'ProPainter/weights/ProPainter.pth':
      ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
        model_dir='weights', progress=True, file_name=None)
    else:
      ckpt_path = args.model
    
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()
    model = model.half()

    
    ##############################################
    # ProPainter inference
    ##############################################
    video_length = frames.size(1)
    print(f'\nProcessing: {video_name} [{video_length} frames]...')
    with torch.no_grad():
        # ---- compute flow ----
        if frames.size(-1) <= 640: 
            short_clip_len = 12
        elif frames.size(-1) <= 720: 
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2
        
        # use fp32 for RAFT
        if video_length > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in trange(0, video_length, short_clip_len, desc='RAFT', leave=args.keep_pbar):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:,f:end_f].cuda(), iters=args.raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames[:,f-1:end_f].cuda(), iters=args.raft_iter)
                
                gt_flows_f_list.append(flows_f.cpu())
                gt_flows_b_list.append(flows_b.cpu())
                
                del flows_f, flows_b; empty_cache()
                
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
            del gt_flows_f, gt_flows_b, gt_flows_f_list, gt_flows_b_list
        else:
            _f, _b = fix_raft(frames.cuda(), iters=args.raft_iter)
            gt_flows_bi = (_f.cpu(), _b.cpu())
            del _f, _b
        empty_cache()

        if args.vram_stat:
            print(f'  Peak allocated: {round(max_memory_allocated()/1024**3, 2)} GB.', f' Peak reserved: {round(max_memory_reserved()/1024**3, 2)} GB.')    
            reset_peak_memory_stats()

        
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > args.subvideo_length:

            pred_flows_f = empty([1, video_length-1, 2,h,w])
            pred_flows_b = empty([1, video_length-1, 2,h,w])
            pad_len = 5
            for f in trange(0, flow_length, args.subvideo_length, desc='Flow completion', leave=args.keep_pbar):

                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + args.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + args.subvideo_length)

                _flow_f = gt_flows_bi[0][:, s_f:e_f]
                _flow_b = gt_flows_bi[1][:, s_f:e_f]
                _flow_m = flow_masks[:, s_f:e_f+1]
                
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (_flow_f, _flow_b), _flow_m)
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (_flow_f, _flow_b), pred_flows_bi_sub, _flow_m)

                _e = e_f - s_f - pad_len_e
                pred_flows_f[:, f:f+_e-pad_len_s] = pred_flows_bi_sub[0][:, pad_len_s:_e]
                pred_flows_b[:, f:f+_e-pad_len_s] = pred_flows_bi_sub[1][:, pad_len_s:_e]

            pred_flows_bi = (pred_flows_f, pred_flows_b)
            del pred_flows_f, pred_flows_b
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)

        if args.vram_stat:
            print(f'  Peak allocated: {round(max_memory_allocated()/1024**3, 2)} GB.', f' Peak reserved: {round(max_memory_reserved()/1024**3, 2)} GB.')    
            reset_peak_memory_stats()


        # ---- image propagation ----
        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, args.subvideo_length) # ensure a minimum of 100 frames for image propagation
        if video_length > subvideo_length_img_prop:
            
            updated_frames = empty([1, video_length, 3,h,w])
            updated_masks = empty([1, video_length, 1,h,w])
            pad_len = 10
            for f in trange(0, video_length, subvideo_length_img_prop, desc='Image propagation', leave=args.keep_pbar):

                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)
                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()

                pred_flows_bi_sub = (
                    pred_flows_bi[0][:, s_f:e_f-1].cuda().half(), 
                    pred_flows_bi[1][:, s_f:e_f-1].cuda().half())

                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(
                    masked_frames[:, s_f:e_f].cuda().half(), pred_flows_bi_sub, 
                    masks_dilated[:, s_f:e_f].cuda().half(), 'nearest')
                
                updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                    prop_imgs_sub.cpu().view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                
                updated_masks_sub = updated_local_masks_sub.cpu().view(b, t, 1, h, w)
                
                _e = e_f - s_f - pad_len_e
                updated_frames[:, f:f+_e-pad_len_s] = updated_frames_sub[:, pad_len_s:_e]
                updated_masks[:, f:f+_e-pad_len_s] = updated_masks_sub[:, pad_len_s:_e]
                
                del pred_flows_bi_sub, prop_imgs_sub, updated_local_masks_sub, updated_frames_sub; empty_cache()
        else:
            b, t, _, _, _ = masks_dilated.size()
            
            prop_imgs, updated_local_masks = model.img_propagation(
                masked_frames.cuda().half(), 
                (pred_flows_bi[0].cuda().half(), pred_flows_bi[1].cuda().half()), 
                masks_dilated.cuda().half(), 'nearest')
            
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.cpu().view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.cpu().view(b, t, 1, h, w)

            del prop_imgs, updated_local_masks; empty_cache()

    if args.vram_stat:
        print(f'  Peak allocated: {round(max_memory_allocated()/1024**3, 2)} GB.', f' Peak reserved: {round(max_memory_reserved()/1024**3, 2)} GB.')    
        reset_peak_memory_stats()


    comp_frames = [None] * video_length

    neighbor_stride = args.neighbor_length // 2
    if video_length > args.subvideo_length:
        ref_num = args.subvideo_length // args.ref_stride
    else:
        ref_num = -1

    ori_flow_masks, ori_masks_dilated = read_mask(args.mask, frames_len, (1280,720), 
                                              flow_mask_dilates=args.mask_dilation,
                                              mask_dilates=args.mask_dilation)
    ori_masks_dilated = to_tensors()(ori_masks_dilated).unsqueeze(0)


    for f in trange(args.skip_frame, video_length, neighbor_stride, desc='Feature propagation + transformer', leave=args.keep_pbar):
        neighbor_ids = [
            i for i in range(max(0, f - neighbor_stride),
                                min(video_length, f + neighbor_stride + 1))
        ]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)

        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (
            pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
            pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])

        with torch.no_grad():
            # 1.0 indicates mask
            l_t = len(neighbor_ids)
            
            # pred_img = selected_imgs # results of image propagation
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.permute(0, 2, 3, 1).numpy() * 255

            ori_binary_masks = ori_masks_dilated[0, neighbor_ids, :, :, :].permute(0, 2, 3, 1).numpy().astype(np.uint8)

            for i in range(l_t):
                idx = neighbor_ids[i]

                pred_img_pil = Image.fromarray(pred_img[i].astype('uint8'))
                pred_img_pil = pred_img_pil.resize((1280,720), Image.LANCZOS)
                pred_img_up = np.array(pred_img_pil).astype(np.uint8) 
                
                img = pred_img_up * ori_binary_masks[i] \
                    + ori_reso_frames[idx] * (1 - ori_binary_masks[i])

                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else: 
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                    
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)


    print(f'  Peak allocated: {round(max_memory_allocated()/1024**3, 2)} GB.', f' Peak reserved: {round(max_memory_reserved()/1024**3, 2)} GB.')    
    reset_peak_memory_stats()


    if args.save_frames:
        for idx in trange(args.skip_frame, video_length, leave=False, desc='saving results'):
            f = comp_frames[idx]
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img_save_root = os.path.join(save_root, str(idx).zfill(4)+'.png')
            imwrite(f, img_save_root)
#        print(f'All results are saved in {save_root}.')

    if args.record_time:
        print(f'Took {round((time()-st)/60, 2)} minutes')
    empty_cache()
