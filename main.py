"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import glob
import os
import numpy as np
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
import data
from utils import weights_init, compute_acc, decimate
from network import _netG, _netD, _netD_CIFAR10, _netG_CIFAR10
from folder import ImageFolder

import visdom
from scipy import misc

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar | imagenet')
parser.add_argument('--data_root', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--train_batch_size', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')

parser.add_argument('--host', default='http://ramawks69', type=str, help="hostname/server visdom listener is on.")
parser.add_argument('--port', default=8097, type=int, help="which port visdom should use.")
parser.add_argument('--visdom_board', default='main', type=str, help="name of visdom board to use.")
parser.add_argument('--eval_period', type=int, default=100)
parser.add_argument('--marygan', type=bool, default=False)

# for data loader
parser.add_argument('--size_labeled_data',  type=int, default=4000)
parser.add_argument('--dev_batch_size',  type=int, default=None)
parser.add_argument('--train_batch_size_2',  type=int, default=None)

opt = parser.parse_args()
print(opt)

# setup visdom
vis = visdom.Visdom(env=opt.visdom_board, port=opt.port, server=opt.host)
visdom_visuals_ids = []
empty_img = np.moveaxis(misc.imread('404_32.png'),-1,0) / 255.0
def winid():
    """
    Pops first item on visdom_visuals_ids or returns none if it is empty
    """
    visid = None # acceptable id to make a new plot
    if visdom_visuals_ids:
        visid = visdom_visuals_ids.pop(0)
    return visid

# specify the gpu id if using only 1 gpu
if opt.ngpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

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

# some hyper parameters
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
num_classes = int(opt.num_classes)
nc = 3
if not opt.dev_batch_size:
    opt.dev_batch_size = 2*opt.train_batch_size
if not opt.train_batch_size_2:
    opt.train_batch_size_2 = opt.train_batch_size

# dataset
if opt.dataset == 'cifar':
    labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set = data.get_cifar_loaders(opt)
else:
    raise NotImplementedError("No such dataset {}".format(opt.dataset))

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
if opt.dataset == 'imagenet':
    netG = _netG(ngpu, nz)
else:
    netG = _netG_CIFAR10(ngpu, nz)
netG.apply(weights_init)
if opt.netG != '':
    print('Loading %s...' % opt.netG)
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# Define the discriminator and initialize the weights
if opt.dataset == 'imagenet':
    netD = _netD(ngpu, num_classes)
else:
    netD = _netD_CIFAR10(ngpu, num_classes)
netD.apply(weights_init)
if opt.netD != '':
    print('Loading %s...' % opt.netD)
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# loss functions
dis_criterion = nn.BCELoss()
aux_criterion = nn.NLLLoss()

# tensor placeholders
input = torch.FloatTensor(opt.train_batch_size, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.train_batch_size, nz, 1, 1)
eval_noise = torch.FloatTensor(opt.train_batch_size, nz, 1, 1).normal_(0, 1)
dis_label = torch.FloatTensor(opt.train_batch_size)
aux_label = torch.LongTensor(opt.train_batch_size)
real_label = 1
fake_label = 0

# if using cuda
if opt.cuda:
    netD.cuda()
    netG.cuda()
    dis_criterion.cuda()
    aux_criterion.cuda()
    input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
    noise, eval_noise = noise.cuda(), eval_noise.cuda()

# define variables
input = Variable(input)
noise = Variable(noise)
eval_noise = Variable(eval_noise)
dis_label = Variable(dis_label)
aux_label = Variable(aux_label)
# noise for evaluation
eval_noise_ = np.random.normal(0, 1, (opt.train_batch_size, nz))
eval_label = np.random.randint(0, num_classes, opt.train_batch_size)
eval_onehot = np.zeros((opt.train_batch_size, num_classes))
eval_onehot[np.arange(opt.train_batch_size), eval_label] = 1
eval_noise_[np.arange(opt.train_batch_size), :num_classes] = eval_onehot[np.arange(opt.train_batch_size)]
eval_noise_ = (torch.from_numpy(eval_noise_))
eval_noise.data.copy_(eval_noise_.view(opt.train_batch_size, nz, 1, 1))

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

avg_loss_D = 0.0
avg_loss_G = 0.0
avg_loss_A = 0.0
history = []
curr_iter = 0
while True:

    curr_iter += 1
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
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
    dis_output, aux_output = netD(input)

    dis_errD_real = dis_criterion(dis_output, dis_label)
    aux_errD_real = aux_criterion(aux_output, aux_label)
    errD_real = dis_errD_real + aux_errD_real
    errD_real.backward()
    D_x = dis_output.data.mean()

    # compute the current classification accuracy
    accuracy = compute_acc(aux_output, aux_label)

    # train with fake
    noise.data.resize_(batch_size, nz, 1, 1).normal_(0, 1)
    label = np.random.randint(0, num_classes, batch_size)
    noise_ = np.random.normal(0, 1, (batch_size, nz))
    if not opt.marygan:
        class_onehot = np.zeros((batch_size, num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
    noise_ = (torch.from_numpy(noise_))
    noise.data.copy_(noise_.view(batch_size, nz, 1, 1))
    aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

    fake = netG(noise)
    dis_label.data.fill_(fake_label)
    dis_output, aux_output = netD(fake.detach())
    dis_errD_fake = dis_criterion(dis_output, dis_label)
    if opt.marygan:
        errD_fake = dis_errD_fake
    else:
        aux_errD_fake = aux_criterion(aux_output, aux_label)
        errD_fake = dis_errD_fake + aux_errD_fake
    errD_fake.backward()
    D_G_z1 = dis_output.data.mean()
    
    # train with unlabeled
    unl_images, _ = unlabeled_loader.next()
    if opt.cuda:
        unl_images = unl_images.cuda()
    input.data.copy_(unl_images)
    dis_label.data.fill_(real_label)
    dis_output, aux_output = netD(input)
    dis_errD_unl = dis_criterion(dis_output, dis_label)
    dis_errD_unl.backward()

    errD = errD_real + errD_fake + dis_errD_unl
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    dis_label.data.fill_(real_label)  # fake labels are real for generator cost
    dis_output, aux_output = netD(fake)
    dis_errG = dis_criterion(dis_output, dis_label)
    if opt.marygan:
        aux_errG = -torch.mean(aux_output.max(1)[0])
    else:
        aux_errG = aux_criterion(aux_output, aux_label)
    errG = dis_errG + aux_errG
    errG.backward()
    D_G_z2 = dis_output.data.mean()
    optimizerG.step()

    # compute the average loss
    all_loss_G = avg_loss_G * curr_iter
    all_loss_D = avg_loss_D * curr_iter
    all_loss_A = avg_loss_A * curr_iter
    all_loss_G += errG.data[0]
    all_loss_D += errD.data[0]
    all_loss_A += accuracy
    avg_loss_G = all_loss_G / (curr_iter + 1)
    avg_loss_D = all_loss_D / (curr_iter + 1)
    avg_loss_A = all_loss_A / (curr_iter + 1)

    print('[%06d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
          % (curr_iter,
             errD.data[0], avg_loss_D, errG.data[0], avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))
    if curr_iter % opt.eval_period == 0:
        history.append([errD.data[0].item(), errG.data[0].item()])
        # setup
        nphist = np.array(history)
        max_line_samples = 200
        ds = max(1,nphist.shape[0] // (max_line_samples+1))
        ts = list(range(opt.eval_period,curr_iter+1,opt.eval_period))
        dts = decimate(ts,ds)

        new_ids = []

        fake = netG(eval_noise)

        real_grid = vutils.make_grid(real_cpu, nrow=10, padding=2, normalize=True)
        new_ids.append(vis.image(real_grid, win=winid(), opts={'title': 'Real Images' }))

        dis_output, aux_output = netD(real_cpu)
        # sort the real images
        real_data = real_cpu.data.cpu()
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
        new_ids.append(vis.image(real_grid_sorted, win=winid(), opts={'title': 'Sorted Real Images' }))

        # fake images
        fake_grid = vutils.make_grid(fake.data, nrow=10, padding=2, normalize=True)
        new_ids.append(vis.image(fake_grid, win=winid(), opts={'title': 'Fixed Fakes' }))

        dis_output, aux_output = netD(fake)
        # same images but sorted
        fixed_fake = fake.data.cpu()
        preds = torch.max(aux_output.detach(),1)[1].data.cpu().numpy()
        sorted_i = np.argsort(preds)
        sorted_preds = preds[sorted_i]
        sorted_imgs = np.zeros(fixed_fake.shape)
        plab = 0
        for i in range(fixed_fake.shape[0]):
            now_lab = i // 10
            # if we have too many images from earlier labels: fast forward
            while (plab < len(sorted_preds)) and (sorted_preds[plab] < now_lab):
                plab += 1

            if (plab == len(sorted_preds)) or (sorted_preds[plab] > now_lab): # ran out of images for this label
                # put in a blank image
                sorted_imgs[i,:,:,:] = empty_img
            elif sorted_preds[plab] == now_lab: # have an image
                # use the image
                sorted_imgs[i,:,:,:] = fixed_fake[sorted_i[plab],:,:,:]
                plab += 1

        # plot sorted fakes
        fake_grid_sorted = vutils.make_grid(torch.Tensor(sorted_imgs), nrow=10, padding=2, normalize=True)
        new_ids.append(vis.image(fake_grid_sorted, win=winid(), opts={'title': 'Sorted Fixed Fakes' }))


        new_ids.append(vis.line(decimate(nphist[:,:2],ds), dts, win=winid(), opts={'legend': ['D', 'G'], 'title': 'Loss'}))

        # vutils.save_image(
        #     real_cpu, '%s/real_samples.png' % opt.outf)
        # print('Label for eval = {}'.format(eval_label))
        # vutils.save_image(
        #     fake.data,
        #     '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch)
        # )

        # done plotting, update ids
        visdom_visuals_ids = new_ids

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_iter_%d.pth' % (opt.outf, curr_iter))
        torch.save(netD.state_dict(), '%s/netD_iter_%d.pth' % (opt.outf, curr_iter))
