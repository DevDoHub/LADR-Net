# LADR-Net

## Installation and Datasets

We use python version 3.7, PyTorch version 1.7.1, CUDA 10.1 and torchvision 0.8.2. More details of installation and dataset preparation can be found in [TransReID-SSL](https://github.com/damo-cv/TransReID-SSL).

## Prepare Pre-trained Models 

```bash
python convert_model.py path/to/SOLIDER/log/lup/swin_tiny/checkpoint.pth path/to/SOLIDER/log/lup/swin_tiny/checkpoint_tea.pth
```


## Training

We utilize 1 GPU for training. Please modify the `MODEL.PRETRAIN_PATH`, `DATASETS.ROOT_DIR` and `OUTPUT_DIR` in the config file.

```bash
sh run.sh
```

## Test

```bash
sh runtest.sh
```

## Performance
