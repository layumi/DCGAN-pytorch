from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import functools

nz = 100 #int(opt.nz)
ngf = 64 #int(opt.ngf)
ndf = 32 #int(opt.ndf)
nc = 3

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('Batchnorm') !=-1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ResnetBlock(nn.Module):
    def __init__(self, fin):
        super().__init__()
        # Submodules
        self.conv_1 = nn.Sequential( nn.Conv2d(fin, fin, 3, stride=1, padding=1),
                      nn.BatchNorm2d(fin),  nn.LeakyReLU(0.2, inplace=True))
        self.conv_2 = nn.Sequential( nn.Conv2d(fin, fin, 3, stride=1, padding=1),
                      nn.BatchNorm2d(fin) )

    def forward(self, x):
        dx = self.conv_1(x)
        dx = self.conv_2(dx)
        out = x + 0.1*dx
        return out

class _netG(nn.Module):
    def __init__(self, ngpu=1, norm_layer=nn.BatchNorm2d):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        use_bias = norm_layer==nn.InstanceNorm2d
        self.main = []
            # input is Z, going into a convolution
        self.main += [nn.ConvTranspose2d( nz, ngf * 8, (4,2), 1, 0, bias=use_bias),
            norm_layer(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True)]
        in_dim = ngf*8
        for i in range(5):
            # state size. (ngf*8) x 4 x 2
            out_dim = max(in_dim//2, 64) 
            self.main += [nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1, bias=use_bias)]
            self.main += [ResnetBlock(out_dim)]

            self.main +=[norm_layer(out_dim),
            nn.LeakyReLU(0.2, inplace=True)]
            in_dim = out_dim

        self.main +=  [nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=0, bias=use_bias),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, nc, kernel_size=3, padding=0, bias=use_bias),
            nn.Tanh()
        ]
        self.main = nn.Sequential(*self.main)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self, ngpu=1, use_sigmoid=True, norm_layer=nn.BatchNorm2d):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func==nn.InstanceNorm2d
        else:
            use_bias = norm_layer==nn.InstanceNorm2d
        sequence = [
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True)]
        sequence += [
            # state size. (ndf) x 32 x 32
            ResnetBlock(ndf),
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            ResnetBlock(ndf * 2),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, 1, kernel_size=3, stride=2, padding=1, bias=use_bias)
        ]
        if use_sigmoid: # classification
            sequence += [nn.Sigmoid()]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class _netE(nn.Module):
    def __init__(self, ngpu=1, use_sigmoid=True, norm_layer=nn.BatchNorm2d):
        super(_netE, self).__init__()
        self.ngpu = ngpu
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func==nn.InstanceNorm2d
        else:
            use_bias = norm_layer==nn.InstanceNorm2d
        sequence = [
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 100, kernel_size=4, stride=1, padding=0, bias=use_bias),
            #nn.Sigmoid()
        ]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
        
