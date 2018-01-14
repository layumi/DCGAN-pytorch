import argparse
import os
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from model import _netG

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='baseline', type=str, help='trained model name')
parser.add_argument('--which_epoch', default='24', type=str, help='0,1,2,3,4...')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--gpu_ids',default='2', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')

opt = parser.parse_args()

str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)
# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])


try:
    os.makedirs(os.path.join('./result',opt.name))
except OSError:
    pass

#---------------generate images
def generate_img(model):
    for i in range(2):
        input_noise = torch.FloatTensor(opt.batchsize, 100, 1, 1).normal_(0,1)
        input_noise = input_noise.cuda()
        input_noise = Variable(input_noise)
        outputs = model(input_noise)
        fake = outputs.data
        print(fake.shape)
        for j in range(opt.batchsize):
            im = fake[j,:,:,:]
            torchvision.utils.save_image(
                    im.view(1, im.size(0), im.size(1), im.size(2)),
                    os.path.join('./result', opt.name, '%d_%d.png'%(i,j)),
                    nrow=1,
                    padding=0,
                    normalize=True)

#-----------------Load model
def load_network(network):
    save_path = os.path.join('./model',opt.name,'netG_epoch_%s.pth'%opt.which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


model_structure = _netG()
model = load_network(model_structure)
model = model.cuda()

generate_img(model)
