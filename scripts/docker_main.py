import os
from os import system, makedirs
from pathlib import Path
from glob import glob
from math import ceil
from concurrent.futures import ThreadPoolExecutor as TPE  # IO bound
from time import time

from PIL import Image
from tqdm import trange, tqdm
from numpy import empty, array, uint8
from SimpleITK import ReadImage, GetImageFromArray, GetArrayFromImage, WriteImage



def calc_indices(n_frames, n_workers):
    ind = []
    start, end = 0, 0
    frames_per_split = ceil(n_frames / n_workers)

    while end < n_frames:
        end += frames_per_split
        if end > n_frames: end = n_frames 
        ind.append((start, end))
        start += frames_per_split
    
    return ind
    

def preprocess(mha, out_dir, start, end, desc='mha to png'):  # out_dir should end with /
    for i in trange(start, end, desc=desc, leave=False):
        Image.fromarray(mha[i]).save(out_dir + str(i).zfill(4) + '.png')

def load_img(result, paths, start, end):
    for i in trange(start, end, desc='loading png', leave=False):
        result[i] = array(Image.open(paths[i]))


get_scene_id = lambda fpath: fpath.split("/")[-1].split(".")[0].split("_")[-1]


def write_array_as_mha(location, scene_id, array):
    """
    The input array should have the shape (num_frames, height, width, channels).
    Individual images should be of UINT type in range [0, 255].
    """
    if not os.path.exists(location): makedirs(location)
    WriteImage(GetImageFromArray(array), os.path.join(location, f"{scene_id}.mha")) #useCompression=True)


MASK_MAP = {
    "6c0da39e-2117-4da4-94dd-ba5e2f8188b7": "dcda170d-a3c0-4fd2-af4a-becc2722b3dd", # scene 0092
    "75904266-551a-4985-85d0-8d414151780b": "44c40c43-f866-443b-a848-647e950aebfe", # scene 0096
    "3497b69b-3f26-46da-baa2-45adce1f8a36": "c4c8c33f-139d-4259-864a-4f2b94651216", # scene 100
    "3e39792c-05fe-4c1e-afa5-6135ea84dd6d": "603924b2-c999-4444-a7b0-f6901404b09b", # scene 101
    "defa1da3-9071-44c0-a590-c5df4cb609af": "c5d44730-6a29-4e21-9ca0-1a024603acac", # scene 102
    "53d1f06f-f0bc-4300-a976-d8f38efcb209": "b4021bd9-9117-4a27-9454-5d577fb4d9cc", # scene 103
    "129b1566-55b3-4951-bfaf-3ca4984baa01": "69e8712d-d2db-46ca-88bb-7d81891b1610", # scene 104
    "c28dcfc3-9247-4008-92a2-936717dae3e3": "240d2cec-c040-4406-933d-0d7fe9f4d4cf", # scene 105
    "b114d8c9-512a-4c22-a94d-0acfe64793b8": "93a142a9-96f1-4b3d-b6cc-78ec43250500", # scene 106
    "979c2c13-ebdb-44f6-8225-68b9e6cd8530": "9083c134-f67d-4d15-a9e4-ca3bde30bbc1", # scene 107
    "6ba52586-8517-4ca1-9fde-fb1701f07bc7": "72507319-66b6-4a90-980c-63a5ca2e2269", # scene 108
    "9cc48058-ddee-4596-aa6c-51836859b989": "28075260-e4c2-407d-96ef-fdf5a154c5a7", # scene 109
}
WORKERS = 8


def main(in_path):
    _t = time()

    scene_ID = get_scene_id(in_path)
    video = GetArrayFromImage(ReadImage(in_path))

    n_frames = len(video)
    indices = calc_indices(n_frames, WORKERS)

    with TPE(WORKERS) as tpe:
        fut = [tpe.submit(preprocess, video, '/tmp/color/', start, end) for start,end in indices]

        mask_path = f'/input/images/synthetic-surgical-scenes-masks/{MASK_MAP[scene_ID]}.mha'
        masks = GetArrayFromImage(ReadImage(mask_path))
        preprocess(masks, '/tmp/mask/', 0, n_frames, 'mask mha to png')

        [_.result() for _ in fut]

    cmd = f'python dynamic_divide.py -n 34 -t 16 -v 350 -r {12 if n_frames>=960 else 15} -s .5 --model weights/728_408_100000.pth -i /tmp/color -m /tmp/mask -o /output'
    system(cmd)

    res = empty((n_frames, 720, 1280, 3), dtype=uint8)
    paths = sorted(glob('/output/result/final_result_dynamic/*png'))

    with TPE(WORKERS) as tpe:
        fut = [tpe.submit(load_img, res, paths, start, end) for start,end in indices]
        [_.result() for _ in fut]

    write_array_as_mha('/output/images/inpainted-synthetic-surgical-scenes', scene_ID, res)

    print(f"{int(time()-_t)/60} minutes")



if __name__ == '__main__':

#    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    Path('/tmp/color').mkdir(); Path('/tmp/mask').mkdir()
    paths = glob('/input/images/synthetic-surgical-scenes/*.mha')

    for p in paths:
        main(p)
