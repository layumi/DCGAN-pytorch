# DCGAN-pytorch (LSGAN)
The code is modified from https://github.com/pytorch/examples/tree/master/dcgan.

## Prerequisites

- Python 3.6
- GPU Memory >= 2G

## Getting started
### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
Because pytorch and torchvision are ongoing projects.

Here we noted that our code is tested based on Pytorch 0.3.0 and Torchvision 0.2.0.

```
mkdir model
mkdir visual
mkdir result
```

### Train
```
python train.py --dataset market --dataroot /home/zzd/market1501/train_pytorch  --withoutE --name baseline-lsgan8x8-encode --lsgan --gpu_ids 3
```

`--name` the name of the output model

`--lsgan` mean using MSELoss(L2Loss)

`--gpu_ids` select which gpu to run

`--batchSize` default 64

by default, batch norm is used. Conv layers have no bias.

`--instance` to use instance norm.  Conv layers also learn bias.

`--withoutE` to remove Encoder Network. The code will run as the basic LSGAN.

Now I set step learning rate schedule. The learning rate drop 0.1 at 40th epoch.

### Test(Generate Images)
```
python test.py  --name baseline  --batchsize 16  --which_epoch 24
```
