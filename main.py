"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import glob
import os
import numpy as np
import random
import re
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import data
from utils import weights_init, compute_acc, decimate, RunningAcc
from network import _netG, _netD, _netD_CIFAR10_SNGAN, _netG_CIFAR10_SNGAN
from folder import ImageFolder
from GAN_training.models import DCGAN, DCGAN_spectralnorm, resnet, resnet_extra, resnet_48_flat, resnet_48, resnet_128

import visdom
import imageio

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar | cifar100')
parser.add_argument('--data_root', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--train_batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--ndf', type=int, default=128)
# parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, ACGAN default=0.0002')
parser.add_argument('--beta1', type=float, default=0.0, help='beta1 for adam. ACGAN default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. ACGAN default=0.999')
parser.add_argument('--ndis', type=int, default=1, help='Num discriminator steps. ACGAN default=1')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--net_type', default='flat', help='Only relevant for image size 48')

parser.add_argument('--host', default='http://ramawks69', type=str, help="hostname/server visdom listener is on.")
parser.add_argument('--port', default=8097, type=int, help="which port visdom should use.")
parser.add_argument('--visdom_board', default='main', type=str, help="name of visdom board to use.")
parser.add_argument('--eval_period', type=int, default=1000)
parser.add_argument('--gantype', default='main', type=str, help="modegan | acgan | mhgan")
parser.add_argument('--scoring_period', type=int, default=10000)
parser.add_argument('--run_scoring_now', type=bool, default=False)
parser.add_argument('--projection_discriminator', type=bool, default=False, help="Set to true to use Miyator Koyama ICLR 2018 projection discriminator")
parser.add_argument('--noise_class_concat', type=bool, default=False, help="Set to true to have class labels writen to the noise")

# for data loader
parser.add_argument('--size_labeled_data',  type=int, default=4000)
parser.add_argument('--dev_batch_size',  type=int, default=None)
parser.add_argument('--train_batch_size_2',  type=int, default=None)

# for GAN_training folder
parser.add_argument('--GAN_lrG', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--GAN_lrD', type=float, default=0.0004, help='learning rate, default=0.0002')
parser.add_argument('--GAN_disc_loss_type', default='hinge', help='which disc loss to use| hinge, wasserstein, ns')

parser.add_argument('--aux_scale_G', type=float, default=0.1, help='WGAN default=0.1')
parser.add_argument('--aux_scale_D', type=float, default=1.0, help='WGAN default=1.0')
parser.add_argument('--max_itr', type=int, default=1e10)
parser.add_argument('--save_period', type=int, default=10000)


opt = parser.parse_args()
print(opt)

# for GAN_training folder
opt.GAN_nz = opt.nz
opt.GAN_ngf = opt.ngf
opt.GAN_ndf = opt.ndf
opt.GAN_disc_iters = opt.ndis
opt.GAN_beta1 = opt.beta1
opt.batchSize = opt.train_batch_size

# setup visdom
vis = visdom.Visdom(env=opt.visdom_board, port=opt.port, server=opt.host)
visdom_visuals_ids = []
if opt.imageSize == 32:
    empty_img = np.moveaxis(imageio.imread('404_32.png')[:,:,:3],-1,0) / 255.0
elif opt.imageSize == 64:
    empty_img = np.moveaxis(imageio.imread('404_64.png')[:,:,:3],-1,0) / 255.0
elif opt.imageSize == 48:
    empty_img = np.moveaxis(imageio.imread('404_48.png')[:,:,:3],-1,0) / 255.0
elif opt.imageSize == 128:
    empty_img = np.moveaxis(imageio.imread('404_128.png')[:,:,:3],-1,0) / 255.0

def winid():
    """
    Pops first item on visdom_visuals_ids or returns none if it is empty
    """
    visid = None # acceptable id to make a new plot
    if visdom_visuals_ids:
        visid = visdom_visuals_ids.pop(0)
    return visid

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
device = torch.device("cuda:0" if opt.cuda else "cpu")

# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = int(opt.num_classes)
num_real_classes = opt.num_classes
if (opt.gantype == 'marygan') or (opt.gantype == 'mhgan'):
    num_real_classes = opt.num_classes - 1
nc = 3
if not opt.dev_batch_size:
    opt.dev_batch_size = opt.train_batch_size
if not opt.train_batch_size_2:
    opt.train_batch_size_2 = opt.train_batch_size

# dataset
if opt.dataset == 'cifar':
    metaloader = data.get_cifar_loaders
elif opt.dataset == 'cifar100':
    metaloader = data.get_cifar100_loaders
elif opt.dataset == 'cifar20':
    metaloader = data.get_cifar20_loaders    
elif opt.dataset == 'stl':
    metaloader = data.get_stl_loaders
elif opt.dataset == 'celeba':
    metaloader = data.get_celeba_loaders
elif opt.dataset == 'flower':
    metaloader = data.get_flower102_loaders
else:
    raise NotImplementedError("No such dataset {}".format(opt.dataset))
labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set = metaloader(opt)

# check outf for files
netGfiles = glob.glob(os.path.join(opt.outf, 'netG_iter_*.pth'))
netGfiles.sort(key = lambda s: int(s.split('_')[-1].split('.')[0]))
if opt.netG == '' and netGfiles:
    opt.netG = netGfiles[-1]

netDfiles = glob.glob(os.path.join(opt.outf, 'netD_iter_*.pth'))
netDfiles.sort(key = lambda s: int(s.split('_')[-1].split('.')[0]))
if opt.netD == '' and netDfiles:
    opt.netD = netDfiles[-1]

# Define the generator and initialize the weights
if opt.imageSize == 32:
    netG = resnet.Generator(opt)
elif opt.imageSize == 64:
    if opt.gantype == 'mhgan' and opt.projection_discriminator:
        netG = resnet_128.Generator64(num_classes=num_real_classes)
    elif opt.gantype == 'mhgan':
        netG = resnet_128.Generator64(num_classes=0)
    else:
        NotImplementedError()
elif opt.imageSize == 48:
    netG = resnet_48.Generator(opt)
elif opt.imageSize == 128:
    if opt.gantype == 'mhgan':
        netG = resnet_128.Generator(num_classes=num_real_classes)
    else:
        NotImplementedError()
else:
    raise NotImplementedError('A network for imageSize %i is not implemented!' % opt.imageSize)
# netG.apply(weights_init)
if opt.netG != '':
    print('Loading %s...' % opt.netG)
    netG.load_state_dict(torch.load(opt.netG))
    curr_iter = int(re.findall('\d+', opt.netG)[-1])
else:
    curr_iter = 0
print(netG)

# Define the discriminator and initialize the weights
if opt.imageSize == 32:
    if (opt.gantype == 'marygan'):
        netD = resnet.Classifier(opt)
    elif opt.gantype == 'acgan':
        netD = resnet.Discriminator(opt)
    elif opt.gantype == 'mhgan':
        netD = resnet.ClassifierMultiHinge(opt)
elif opt.imageSize == 64:
    if opt.gantype == 'mhgan' and opt.projection_discriminator:
        netD = resnet_128.ClassifierMultiHinge64(num_classes=opt.num_classes)
    elif opt.gantype == 'mhgan':
        netD = resnet_128.ClassifierMultiHinge64(num_classes=opt.num_classes)
    else:
        NotImplementedError()
elif opt.imageSize == 48:
    if opt.gantype == 'mhgan':
        netD = resnet_48.ClassifierMultiHinge(opt)
    else:
        NotImplementedError()
elif opt.imageSize == 128:
    if opt.gantype == 'mhgan':
        netD = resnet_128.ClassifierMultiHinge(num_classes=opt.num_classes)
    else:
        NotImplementedError()
else:
    raise NotImplementedError('A network for imageSize %i is not implemented!' % opt.imageSize)
# netD.apply(weights_init)
if opt.netD != '':
    print('Loading %s...' % opt.netD)
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# loss functions
nll = nn.NLLLoss()
def dis_criterion(inputs, labels):
    # hinge loss
    # from Yogesh, probably from: https://github.com/wronnyhuang/gan_loss/blob/master/trainer.py
    return torch.mean(F.relu(1 + inputs*labels)) + torch.mean(F.relu(1 - inputs*(1-labels)))
# dis_criterion = nn.BCELoss()
acgan_aux_criterion = nll
def marygan_criterion(inputs, labels):
    return nll(torch.log(inputs),labels)
def crammer_singer_criterion(X, Ylabel):
    mask = torch.ones_like(X)
    mask.scatter_(1, Ylabel.unsqueeze(-1), 0)
    wrongs = torch.masked_select(X,mask.byte()).reshape(X.shape[0],opt.num_classes-1)
    max_wrong, _ = wrongs.max(1)
    max_wrong = max_wrong.unsqueeze(-1)
    target = X.gather(1,Ylabel.unsqueeze(-1))
    return torch.mean(F.relu(1 + max_wrong - target))
def crammer_singer_complement_criterion(X, Ylabel):
    mask = torch.ones_like(X)
    mask.scatter_(1, Ylabel.unsqueeze(-1), 0)
    wrongs = torch.masked_select(X,mask.byte()).reshape(X.shape[0],opt.num_classes-1)
    max_wrong, _ = wrongs.max(1)
    max_wrong = max_wrong.unsqueeze(-1)
    target = X.gather(1,Ylabel.unsqueeze(-1))
    return torch.mean(F.relu(1 - max_wrong + target))
def hinge_antifake(X, Ylabel):
    target = X.gather(1,Ylabel.unsqueeze(-1))
    label_for_fake = (opt.num_classes - 1) * np.ones(Ylabel.shape[0])
    Ylabel.data.copy_(torch.from_numpy(label_for_fake))
    pred_for_fake = X.gather(1,Ylabel.unsqueeze(-1))
    return torch.mean(F.relu(1 + pred_for_fake - target))

if (opt.gantype == 'marygan'):
    aux_criterion = marygan_criterion
elif opt.gantype == 'acgan':
    aux_criterion = acgan_aux_criterion
elif opt.gantype == 'mhgan':
    aux_criterion = crammer_singer_criterion


# tensor placeholders
input = torch.FloatTensor(opt.train_batch_size, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.train_batch_size, nz)
eval_noise = torch.FloatTensor(opt.train_batch_size, nz).normal_(0, 1)
dis_label = torch.FloatTensor(opt.train_batch_size)
aux_label = torch.LongTensor(opt.train_batch_size) # use in criterion
conditioning_label = torch.LongTensor(opt.train_batch_size) # use as network input
eval_conditioning_label = torch.LongTensor(opt.train_batch_size)
real_label = 1
fake_label = 0
K = opt.num_classes
if (opt.gantype == 'marygan'):
    K = opt.num_classes - 1

# if using cuda
if opt.cuda:
    netG = netG.to(device)
    netD = netD.to(device)
    # dis_criterion.cuda()
    # aux_criterion.cuda()
    input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
    noise, eval_noise = noise.cuda(), eval_noise.cuda()
    conditioning_label = conditioning_label.cuda()
    eval_conditioning_label = eval_conditioning_label.cuda()

# define variables
input = Variable(input)
noise = Variable(noise)
eval_noise = Variable(eval_noise)
dis_label = Variable(dis_label)
aux_label = Variable(aux_label)
conditioning_label = Variable(conditioning_label)
eval_conditioning_label = Variable(eval_conditioning_label)
# noise for evaluation
eval_noise_ = np.random.normal(0, 1, (opt.train_batch_size, nz))
eval_label_ = np.random.randint(0, num_classes, opt.train_batch_size)
eval_onehot = np.zeros((opt.train_batch_size, num_classes))
eval_onehot[np.arange(opt.train_batch_size), eval_label_] = 1
eval_noise_[np.arange(opt.train_batch_size), :num_classes] = eval_onehot[np.arange(opt.train_batch_size)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(opt.train_batch_size, nz))
eval_conditioning_label.data.copy_(torch.from_numpy(eval_label_))

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.GAN_lrD, betas=(opt.beta1, opt.beta2))
optimizerG = optim.Adam(netG.parameters(), lr=opt.GAN_lrG, betas=(opt.beta1, opt.beta2))


avg_loss_D = 0.0
avg_loss_G = 0.0
avg_loss_A = 0.0
history = []
history_times = []
score_history = []
score_history_times = []

latest_save = None
this_run_iters = 0
this_run_seconds = 0
running_accuracy = RunningAcc(100)


while curr_iter <= opt.max_itr:
    curr_iter += 1
    this_run_iters +=1
    this_iter_start = time.time()
    for dis_step in range(opt.ndis):
        ############################
        # (1) Update D network
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, label = labeled_loader.next()
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.data.resize_as_(real_cpu).copy_(real_cpu)
        dis_label.data.resize_(batch_size).fill_(real_label)
        aux_label.data.resize_(batch_size).copy_(label)
        conditioning_label.data.resize_(batch_size).copy_(label)
        
        if opt.projection_discriminator:
            dis_output, aux_output = netD(input, conditioning_label)
        else:
            dis_output, aux_output = netD(input)

        if (opt.gantype == 'marygan'):
            aux_errD_real = aux_criterion(aux_output, aux_label)
            errD_real = aux_errD_real
        elif opt.gantype == 'acgan':
            aux_errD_real = aux_criterion(aux_output, aux_label)
            dis_errD_real = dis_criterion(dis_output, dis_label)
            errD_real = dis_errD_real + opt.aux_scale_D * aux_errD_real
        elif opt.gantype == 'mhgan':
            aux_errD_real = aux_criterion(aux_output, aux_label)
            errD_real = aux_errD_real
        errD_real.backward()
        D_x = dis_output.data.mean()

        # compute the current classification accuracy on train
        if dis_step == 0:
            if opt.gantype == 'mhgan':
                accuracy = compute_acc(aux_output[:,:(opt.num_classes-1)], aux_label)
            else:
                accuracy = compute_acc(aux_output, aux_label)

        # train with fake
        label = np.random.randint(0, num_classes, batch_size)
        aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
        conditioning_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
        noise.data.resize_(batch_size, nz).normal_(0, 1)
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        if opt.noise_class_concat:
            class_onehot = np.zeros((batch_size, num_classes))
            class_onehot[np.arange(batch_size), label] = 1
            noise_[np.arange(batch_size), :num_classes] = class_onehot
        if opt.gantype == 'mhgan':
            # overwrite aux_label to signify fake
            label_for_fake = (opt.num_classes - 1) * np.ones(batch_size)
            aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label_for_fake))
        noise_ = (torch.from_numpy(noise_))
        noise.data.copy_(noise_.view(batch_size, nz))
        dis_label.data.fill_(fake_label)

        if opt.projection_discriminator:
            fake = netG(noise, conditioning_label)
            dis_output, aux_output = netD(fake.detach(), conditioning_label)
        else:
            fake = netG(noise)
            dis_output, aux_output = netD(fake.detach())

        if (opt.gantype == 'marygan'):
            errD_fake = -torch.mean(torch.log(dis_output))
        elif opt.gantype == 'acgan':
            dis_errD_fake = dis_criterion(dis_output, dis_label)
            aux_errD_fake = aux_criterion(aux_output, aux_label)
            errD_fake = dis_errD_fake + opt.aux_scale_D * aux_errD_fake
        elif opt.gantype == 'mhgan':
            errD_fake = aux_criterion(aux_output, aux_label)

        errD_fake.backward()
        D_G_z1 = dis_output.data.mean()
        errD = errD_real + errD_fake
        
        # train with unlabeled
        if unlabeled_loader and opt.gantype == 'mhgan':
            unl_images, _ = unlabeled_loader.next()
            if opt.cuda:
                unl_images = unl_images.cuda()
            input.data.copy_(unl_images)
            dis_label.data.fill_(real_label)
            if opt.projection_discriminator:
                dis_output, aux_output = netD(input, conditioning_label)
            else:
                dis_output, aux_output = netD(input)
            
            # dis_errD_unl = dis_criterion(dis_output, dis_label)

            dis_errD_unl = crammer_singer_complement_criterion(aux_output, aux_label)

            dis_errD_unl.backward()
            errD += dis_errD_unl

        optimizerD.step()

    ############################
    # (2) Update G network
    ###########################
    netG.zero_grad()
    dis_label.data.fill_(real_label)  # fake labels are real for generator cost

    if opt.projection_discriminator:
        dis_output, aux_output = netD(fake, conditioning_label)
    else:
        dis_output, aux_output = netD(fake)

    if (opt.gantype == 'marygan'):
        errG = -torch.mean(torch.log(aux_output).max(1)[0])
    elif opt.gantype == 'acgan':
        dis_errG = dis_criterion(dis_output, dis_label)
        aux_errG = aux_criterion(aux_output, aux_label)
        errG = dis_errG + opt.aux_scale_G * aux_errG
    elif opt.gantype == 'mhgan':
        errG = crammer_singer_complement_criterion(aux_output, aux_label)
        
        # aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
        # errG = aux_criterion(aux_output, aux_label)

        # errG = hinge_antifake(aux_output, aux_label)

    errG.backward()
    D_G_z2 = dis_output.data.mean()
    optimizerG.step()

    ############################
    # A little running eval accuracy
    ###########################
    if dev_loader:
        real_cpu, label = dev_loader.next()
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.data.resize_as_(real_cpu).copy_(real_cpu)
        dis_label.data.resize_(batch_size).fill_(real_label)
        aux_label.data.resize_(batch_size).copy_(label)
        dis_output, aux_output = netD(input.detach())
        if opt.gantype == 'mhgan':
            aux_output = aux_output[:,:(opt.num_classes-1)]

        test_accuracy, test_acc_dev = running_accuracy.compute_acc(aux_output, aux_label)
    else:
        test_accuracy, test_acc_dev = -1, -1

    # compute the average loss
    all_loss_G = avg_loss_G * this_run_iters
    all_loss_D = avg_loss_D * this_run_iters
    all_loss_A = avg_loss_A * this_run_iters
    all_loss_G += errG.item()
    all_loss_D += errD.item()
    all_loss_A += accuracy
    avg_loss_G = all_loss_G / (this_run_iters + 1)
    avg_loss_D = all_loss_D / (this_run_iters + 1)
    avg_loss_A = all_loss_A / (this_run_iters + 1)

    this_run_seconds += (time.time() - this_iter_start)

    print('[%06d][%.2f itr/s] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Train Acc: %.4f (%.4f) Test Acc: %.4f +/- %.2f'
          % (curr_iter, this_run_iters / this_run_seconds,
             errD.item(), avg_loss_D, errG.item(), avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A, test_accuracy, test_acc_dev))
    
    ### Save GAN Images to interface with IS and FID scores
    if opt.run_scoring_now or curr_iter % opt.scoring_period == 0:
        opt.run_scoring_now = False
        n_used_imgs = 50000
        all_fakes = np.empty((n_used_imgs,opt.imageSize,opt.imageSize,3), dtype=np.uint8)
        if not os.path.exists('%s/GAN_OUTPUTS' % (opt.outf)):
            os.makedirs('%s/GAN_OUTPUTS' % (opt.outf))
        # save a bunch of GAN images
        for i in tqdm(range(n_used_imgs // opt.train_batch_size),desc='Saving'):
            start = i * opt.train_batch_size
            end = start + opt.train_batch_size

            # fake
            noise.data.resize_(batch_size, nz).normal_(0, 1)
            label = np.random.randint(0, num_classes, batch_size)
            noise_ = np.random.normal(0, 1, (batch_size, nz))
            if opt.noise_class_concat:
                class_onehot = np.zeros((batch_size, num_classes))
                class_onehot[np.arange(batch_size), label] = 1
                noise_[np.arange(batch_size), :num_classes] = class_onehot
            noise_ = (torch.from_numpy(noise_))
            noise.data.copy_(noise_.view(batch_size, nz))
            fake = netG(noise).data.cpu().numpy()
            
            fake = np.floor((fake + 1) * 255/2.0).astype(np.uint8)
            all_fakes[start:end] = np.moveaxis(fake,1,-1)

        np.save('%s/GAN_OUTPUTS/out.npy' % (opt.outf), all_fakes)
        with open('%s/scoring.info' % opt.outf,'w') as f:
            f.write(str(curr_iter))


    if (curr_iter-1) % opt.eval_period == 0:
        history.append([errD.item(), errG.item()])
        history_times.append(curr_iter)
        # setup
        nphist = np.array(history)
        max_line_samples = 200
        ds = max(1,nphist.shape[0] // (max_line_samples+1))
        dts = decimate(history_times,ds)

        new_ids = []


        real_grid = vutils.make_grid(real_cpu, nrow=10, padding=2, normalize=True)
        new_ids.append(vis.image(real_grid, win=winid(), opts={'title': 'Real Images' }))

        dis_output, aux_output = netD(real_cpu)
        # sort the real images
        real_data = real_cpu.data.cpu()
        if opt.gantype == 'mhgan':
            preds = torch.max(aux_output.detach()[:,:(opt.num_classes-1)],1)[1].data.cpu().numpy()
        else:
            preds = torch.max(aux_output.detach(),1)[1].data.cpu().numpy()
        sorted_i = np.argsort(preds)
        sorted_preds = preds[sorted_i]
        sorted_real_imgs = np.zeros(real_data.shape)
        plab = 0
        for i in range(real_data.shape[0]):
            now_lab = i // 10
            # if we have too many images from earlier labels: fast forward
            while (plab < len(sorted_preds)) and (sorted_preds[plab] < now_lab):
                plab += 1

            if (plab == len(sorted_preds)) or (sorted_preds[plab] > now_lab): # ran out of images for this label
                # put in a blank image
                sorted_real_imgs[i,:,:,:] = empty_img
            elif sorted_preds[plab] == now_lab: # have an image
                # use the image
                sorted_real_imgs[i,:,:,:] = real_data[sorted_i[plab],:,:,:]
                plab += 1

        # plot sorted reals
        real_grid_sorted = vutils.make_grid(torch.Tensor(sorted_real_imgs), nrow=10, padding=2, normalize=True)
        new_ids.append(vis.image(real_grid_sorted, win=winid(), opts={'title': 'Sorted Real Images ' }))

        # fake images
        if opt.projection_discriminator:
            fake = netG(eval_noise, eval_conditioning_label)
        else:
            fake = netG(eval_noise)

        fake_grid = vutils.make_grid(fake.data, nrow=10, padding=2, normalize=True)
        new_ids.append(vis.image(fake_grid, win=winid(), opts={'title': 'Fixed Fakes' }))

        # fake images, but sorted
        n_display_per_class = 10
        n_fakes_to_sort = n_display_per_class * num_real_classes
        n_batches_to_generate = (n_fakes_to_sort // batch_size) + 1
        sorted_fakes_shape = (n_batches_to_generate * batch_size, nc, opt.imageSize, opt.imageSize)
        all_fake_imgs = np.zeros(sorted_fakes_shape)
        all_preds = np.zeros((sorted_fakes_shape[0],))
        # generate these images
        for i in range(n_batches_to_generate):
            # set noise
            label = np.random.randint(0, num_classes, batch_size)
            aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
            noise_ = np.random.normal(0, 1, (batch_size, nz))
            conditioning_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
            if opt.noise_class_concat:
                class_onehot = np.zeros((batch_size, num_classes))
                class_onehot[np.arange(batch_size), label] = 1
                noise_[np.arange(batch_size), :num_classes] = class_onehot
            noise_ = (torch.from_numpy(noise_))
            noise.data.copy_(noise_.view(batch_size, nz))
            # generate
            if opt.projection_discriminator:
                fake = netG(noise, conditioning_label)
                dis_output, aux_output = netD(fake, conditioning_label)
            else:
                fake = netG(noise)
                dis_output, aux_output = netD(fake)
            if opt.gantype == 'mhgan':
                preds = torch.max(aux_output.detach()[:,:(opt.num_classes-1)],1)[1].data.cpu().numpy()
            else:
                preds = torch.max(aux_output.detach(),1)[1].data.cpu().numpy()
            # save
            all_fake_imgs[(i*batch_size):((i+1)*batch_size)] = fake.data.cpu()
            all_preds[(i*batch_size):((i+1)*batch_size)] = preds

        sorted_i = np.argsort(all_preds)
        sorted_preds = all_preds[sorted_i]
        sorted_imgs = np.zeros(sorted_fakes_shape)
        plab = 0 # pointer in sorted arrays
        for i in range(sorted_fakes_shape[0]): # iterate through unfilled sorted_imgs
            now_lab = i // n_display_per_class
            # if we have too many images from earlier labels: fast forward
            while (plab < sorted_fakes_shape[0]) and (sorted_preds[plab] < now_lab):
                plab += 1

            if (plab == sorted_fakes_shape[0]) or (sorted_preds[plab] > now_lab): # ran out of images for this label
                # put in a blank image
                sorted_imgs[i,:,:,:] = empty_img
            elif sorted_preds[plab] == now_lab: # have an image
                # use the image
                sorted_imgs[i,:,:,:] = all_fake_imgs[sorted_i[plab],:,:,:]
                plab += 1

        # plot sorted fakes in groups of classes
        n_classes_display_per_group = 10
        for i in range(num_real_classes // n_classes_display_per_group):
            group = sorted_imgs[(i*n_display_per_class*n_classes_display_per_group):((i+1)*n_display_per_class*n_classes_display_per_group)]
            fake_grid_sorted = vutils.make_grid(torch.Tensor(group), nrow=10, padding=2, normalize=True)
            new_ids.append(vis.image(fake_grid_sorted, win=winid(), opts={'title': 'Sorted Fakes Class %i-%i' % (i*n_classes_display_per_group,(i+1)*n_classes_display_per_group) }))

        new_ids.append(vis.line(decimate(nphist[:,:2],ds), dts, win=winid(), opts={'legend': ['D', 'G'], 'title': 'Loss'}))

        # done plotting, update ids
        visdom_visuals_ids = new_ids

        # save latest
        if latest_save:
            os.remove('%s/netG_iter_%06d.pth' % (opt.outf, latest_save))
            os.remove('%s/netD_iter_%06d.pth' % (opt.outf, latest_save))
        torch.save(netG.state_dict(), '%s/netG_iter_%06d.pth' % (opt.outf, curr_iter))
        torch.save(netD.state_dict(), '%s/netD_iter_%06d.pth' % (opt.outf, curr_iter))
        latest_save = curr_iter

    # do historical checkpointing
    if curr_iter % opt.save_period == (opt.save_period - 1):
        torch.save(netG.state_dict(), '%s/netG_iter_%06d.pth' % (opt.outf, curr_iter))
        torch.save(netD.state_dict(), '%s/netD_iter_%06d.pth' % (opt.outf, curr_iter))
        vutils.save_image(fake.data,'%s/fake_samples_iter_%06d.png' % (opt.outf, curr_iter), normalize=True)

print("Max iteration of %i reached. Quiting..." % opt.max_itr)
