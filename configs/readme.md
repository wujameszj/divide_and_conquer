
## How to train / fine-tune

1. Prepare training data, e.g., `python scripts/resize.py -i data -o temp/512_288 -s 512,288`  
3. If storage allows, pre-compute flow, e.g., `python scripts/compute_flow.py -i temp/512_288 -o temp/flow_512_288 -w 512 -h 288`
4. Specify training config as .json
5. If fine-tuning, place pretrained model (eg. ProPainter.pth) in output directory 
6. `python scripts/train.py -c config.json`


## Training configs

Changes to config provided by ProPainter author:
- reduce batch size, workers, and num_prefetch_queue to 1 to fit in VRAM 
- use pre-computed flow --> 'load_flow': true
- removed all 10 scenes from validation set [12, 16, 35, 38, 39, 44, 75, 90, 95, 96]


### 712_400

712x400p is about the largest that will fit into 48GB VRAM.
Trained on a single GPU.

- reduced lr to 5e-5 from 1e-4 since we are finetuning (instead of training from scratch)


### 728_408

- further reduced initial LR to 1e-5



## Misc

__Error when running tensorboard:__
- `TypeError: MessageToJson() got an unexpected keyword argument 'including_default_value_fields'`  
- [Downgrade protobuf to 4.25:](https://github.com/tensorflow/tensorboard/issues/6808) `pip install protobuf==4.25`
