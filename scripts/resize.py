from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor as TPE  # IO bound
from concurrent.futures import ProcessPoolExecutor as PPE  # cpu bound

from tqdm import tqdm
from PIL import Image



def resize(in_dir, out_dir, new_size):
    
    out_dir = out_dir.joinpath(in_dir.parent.name[-2:])  # scene ID
    out_dir.mkdir(exist_ok=True)
    
    for p in tqdm(sorted(in_dir.glob('*jpg')), desc=in_dir.parent.name, leave=False):
        i = Image.open(p)
        i = i.resize(new_size if new_size else i.size)
        i.save(out_dir.joinpath(p.stem + '.png'))
        
        
        
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='data/', help='Path to content extracted from data zip file.')
    parser.add_argument('-o', '--output', type=str, default='temp/training_data/', help='Output folder.')
    parser.add_argument('-p', '--pattern', type=str, default='scene*/gt', help='glob pattern for input dir')
    parser.add_argument('-s', '--size', type=str, default='', help='New dimensions as w,h or blank if resizing not needed.')
    parser.add_argument('-w', '--workers', type=int, default=28, help='number of workers')
    args = parser.parse_args()

    gt_dirs = sorted(Path(args.input).glob(args.pattern))
    out_dir = Path(args.output); out_dir.mkdir(exist_ok=True)
    new_size = eval(args.size) if args.size else args.size
    
            
    with TPE(args.workers) as tpe, tqdm(total=len(gt_dirs)) as pbar:
        for _ in [tpe.submit(resize, d, out_dir, new_size) for d in gt_dirs]:
            _.result(); pbar.update()



