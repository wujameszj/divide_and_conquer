from os import system
from os.path import isfile

from pathlib import Path
import json
from argparse import ArgumentParser

from fid import FrechetInceptionDistanceMod as FID

from torch import sum, stack, manual_seed
from torchvision.io import read_image
from torchmetrics.regression import MeanAbsoluteError as MAE
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.functional.image.lpips import _NoTrainLpips, _lpips_update, _lpips_compute

from tqdm import trange


'''
Example argument:

    pred_path = '/mnt/ssd1/james/dreaming_video_inpainting/temp/scene0_462_578/results_finetune/pasted_back'
    mask_path = '/mnt/ssd1/james/dreaming_video_inpainting/data/DREAMING_training_0000-0009/scene_0000/mask'
    gt_path = '/mnt/ssd1/james/dreaming_video_inpainting/data/DREAMING_training_0000-0009/scene_0000/gt'
'''
def _eval(pred_path, mask_path, gt_path, device):
    
    _ = manual_seed(123)
    invalid_frame = lambda mask_frame: sum(255 - mask_frame) == 0
    
    png_paths = sorted(Path(pred_path).glob('*png'))
    pred = stack([read_image(str(p)) for p in png_paths])

    mask_paths = sorted(Path(mask_path).glob('*png'))
    masks = stack([read_image(str(p)) for p in mask_paths])

    gt_paths = sorted(Path(gt_path).glob('*jpg'))
    gt = stack([read_image(str(p)) for p in gt_paths])
    
    lpips_net = _NoTrainLpips(net="alex", pretrained=True, pnet_rand=True, model_path="../temp/alexnet.pth")
    fid_net = FID(feature=2048, normalize=True, feature_extractor_weights_path="../temp/inception.pth")
    lpips_net.to(device), fid_net.to(device)

    mae, psnr = MAE().to(device), PSNR().to(device)
    results = {"mae": 0, "psnr": 0, "lpips": 0}
    
    
    valid_frames = 0
    for i in trange(len(gt), leave=False):

        if invalid_frame(masks[i]): continue

        _pred = pred[i:i+1].to(device)   # i:i+1 instead of unsqueeze(0)
        _gt = gt[i:i+1].to(device)  

        results["mae"] += mae(_pred, _gt).item() / 255.0
        results["psnr"] += psnr(_pred, _gt).item()

        pred_norm = (_pred / 255.0).half()
        gt_norm = (_gt / 255.0).half()

        lpips_loss, lpips_total = _lpips_update(pred_norm, gt_norm, net=lpips_net, normalize=True)
        results["lpips"] += _lpips_compute(lpips_loss.sum(), lpips_total, reduction="mean").item()

        fid_net.update(pred_norm, real=False)
        fid_net.update(gt_norm, real=True) 

        valid_frames += 1


    results["mae"] /= valid_frames
#    results["mae"] = results["mae"] / 255.0 * 200.0  # normalize to [0, 1]

    results["psnr"] /= valid_frames
#    results["psnr"] = (1 - results["psnr"] / 50.0)  # normalize to [0, 1]

    results["lpips"] /= valid_frames
#    results["lpips"] = results["lpips"] * 10.0  # weight the lpips

    results["fid"] = fid_net.compute().item()
#    results["fid"] = fid / 100.0

    results['accuracy'] = (results["mae"]/255*200 + 1-results["psnr"]/50) / 2
    results['consistency'] = (results["lpips"]*10 + results["fid"]/100) / 2

    return results  



def _round(dic: dict, ndig=4):
    for k,v in dic.items():
        dic[k] = round(v, ndigits=ndig)
    return dic



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-i', '--pred', type=str, default='', help='directory containing predictions')
    parser.add_argument('-m', '--mask', type=str, default='', help='mask directory')
    parser.add_argument('-g', '--ground_truth', type=str, default='', help='ground truth directory')
    parser.add_argument('-o', '--output', type=str, default='', help='output path')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='cuda device or cpu')
    args = parser.parse_args()

    if not isfile('../temp/alexnet.pth'):
        system('wget https://github.com/cgsaxner/DREAMING-challenge/raw/main/evaluation/resources/alexnet.pth -O ../temp/alexnet.pth')
    if not isfile('../temp/inception.pth'):
        system('wget https://github.com/cgsaxner/DREAMING-challenge/raw/main/evaluation/resources/inception.pth -O ../temp/inception.pth')

    res = _eval(args.pred, args.mask, args.ground_truth, args.device)
    res = _round(res)

    print('scene', args.ground_truth[-5:-3], res)

    if args.output:
        with Path(args.output).open('w') as _:
            json.dump(res, _, indent=2)
