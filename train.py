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
from model import _netG, _netD, _netE, weights_init
from image_pool import ImagePool
from torch.optim import lr_scheduler

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=60, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0, help='beta1 for adam. default=0.5')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--name', default='baseline', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--gpu_ids', default='2', type=str, help='gpu_ids: e.g. 0 0,1,2 0,2')
parser.add_argument('--lsgan', action='store_true', help='use lsgan')
parser.add_argument('--instance', action='store_true', help='use instance norm')
parser.add_argument('--withoutE', action='store_true', help='do not use Encoder Network')

opt = parser.parse_args()
str_ids = opt.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id>=0:
        gpu_ids.append(id)

print(opt)

try:
    os.makedirs(os.path.join('./model',opt.name))
    os.makedirs(os.path.join('./visual',opt.name))
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.cuda=False
if torch.cuda.is_available():
    opt.cuda=True
    torch.cuda.manual_seed_all(opt.manualSeed)
    torch.cuda.set_device(gpu_ids[0])

cudnn.benchmark = True

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'market':
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize((opt.imageSize)),
                                   #transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5)),
                                   ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = len(gpu_ids)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3

#-------Initial Model ----------
#--------E-----------
if opt.instance:
    netG = _netG(ngpu, norm_layer=nn.InstanceNorm2d)
    netG.apply(weights_init)
else:
    netG = _netG(ngpu)
    netG.apply(weights_init)

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

#---------D------------
if opt.instance:
    netD = _netD(ngpu, use_sigmoid=(not opt.lsgan), norm_layer=nn.InstanceNorm2d)
    netD.apply(weights_init)
else:
    netD = _netD(ngpu, use_sigmoid=(not opt.lsgan))
    netD.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

#---------E--------------
if not opt.withoutE:
    if opt.instance:
        netE = _netE(ngpu, use_sigmoid=(not opt.lsgan), norm_layer=nn.InstanceNorm2d)
        netE.apply(weights_init)
    else:
        netE = _netE(ngpu, use_sigmoid=(not opt.lsgan))
        netE.apply(weights_init)
    print(netE)

#----------Loss-----------
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=False, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.lamb = 0.0001
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            real_tensor = self.Tensor(input.size()).fill_(self.real_label)
            self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
            self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, output, target_is_real, is_real=False):
        target_tensor = self.get_target_tensor(output, target_is_real)
        loss =  self.loss(output, target_tensor)
        if is_real:
            reg = self.lamb *self.compute_grad2(output, input).mean()
        else:
            reg = 0
        loss += reg
        return loss

    def compute_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg = grad_dout2.view(batch_size, -1).sum(1)
        return reg


criterion = GANLoss(use_lsgan=opt.lsgan, tensor=torch.cuda.FloatTensor)
criterionL1 = nn.L1Loss()
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize*2, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1) 

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    if not opt.withoutE:
        netE.cuda()
    criterion.cuda()
    criterionL1.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=2*opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
if not opt.withoutE:
    optimizerE = optim.Adam(netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

fake_pool = ImagePool(50)

schedulers = []
schedulers.append(lr_scheduler.StepLR(optimizerD, step_size=20, gamma=0.5))
schedulers.append(lr_scheduler.StepLR(optimizerG, step_size=20, gamma=0.5))
if not opt.withoutE:
    schedulers.append(lr_scheduler.StepLR(optimizerE, step_size=20, gamma=0.5))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        inputv = Variable(input)
        inputv.requires_grad_()  # for regularization
        output = netD(inputv)
        errD_real = criterion(inputv, output, True, is_real=True)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        print(fake.shape)
        #fake_re = fake_pool.query(fake.data)  #For D, we use a replay policy
        output = netD(fake.detach())
        errD_fake = criterion(fake.detach(), output, False)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        output = netD(fake)
        if not opt.withoutE:
            embedding = netE(fake)
            errG = criterion(fake, output, True) + criterionL1(embedding, noisev)
        else:
            errG = criterion(fake, output, True)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()
        
        ###########################
        # (3) Update E
        if not opt.withoutE:
            netE.zero_grad()
            embedding = netE(fake.detach())
            errE = criterionL1(embedding, noisev)
            errE.backward()
            optimizerE.step()
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f  Loss_E: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], errE.data[0], D_x, D_G_z1, D_G_z2))
        else:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f  D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, opt.niter, i, len(dataloader),
                    errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
    ###########################
    # visualize model
    vutils.save_image(real_cpu,
                    './visual/%s/real_samples.png' % opt.name,
                    normalize=True)
    fake = netG(fixed_noise)
    vutils.save_image(fake.data,
                    './visual/%s/fake_samples_epoch_%03d.png' % (opt.name, epoch),
                    normalize=True)

    # do checkpointing
    if epoch%10 == 0:
        torch.save(netG.state_dict(), './model/%s/netG_epoch_%d.pth' % (opt.name, epoch))
        torch.save(netD.state_dict(), './model/%s/netD_epoch_%d.pth' % (opt.name, epoch))
        if not opt.withoutE:
            torch.save(netE.state_dict(), './model/%s/netE_epoch_%d.pth' % (opt.name, epoch))
    #step lrRate
    for scheduler in  schedulers:
        scheduler.step()
