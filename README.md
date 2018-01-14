# DCGAN-pytorch

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


### Train
```
python main.py --dataset market --dataroot /home/zzd/market1501/train_pytorch  --name baseline
```
