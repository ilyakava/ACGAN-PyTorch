from __future__ import absolute_import, division, print_function
import argparse
import os
import re
from collections import defaultdict
import glob
import time
import pathlib
import imageio
import sys
import numpy as np
import fid
import inception as iscore
import imageio
import tensorflow as tf
from torchvision.datasets import CIFAR10, STL10
import torch
import torchvision.utils as vutils
import torch.utils.data as utils
import visdom
from torchvision import transforms
from GAN_training.models import resnet, resnet_extra, resnet_48
from classification.models.vgg_face_dag import vgg_face_dag
from classification.models.densenet import DenseNet121, densenet_stl
from classification.models.vgg_official2 import vgg16
from tqdm import tqdm

import data

from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

import pdb

n_used_imgs = 4000
# xp = '/scratch0/ilya/locDoc/ACGAN/experiments/'
xp = '/fs/vulcan-scratch/ilyak/locDoc/experiments/'


def cifar_plots(mode, klass_picked):
    class optclass:
        workaround = True
    
    
        
    opt = optclass()
    optdict = {
        #'outf': '/scratch0/ilya/locDoc/ACGAN/experiments/yogesh_acgan_0p2',
        'outf': xp+'marygan-stl-48-miyato-hyp-lrGp4-auxp4',
        'netG': xp+'marygan-stl-48-miyato-hyp-lrGp4-auxp4/netG_iter_069999.pth',
        'marygan': True,
        'imageSize': 48,
        'data_root': '',
        'dataset': 'cifar',
        'dev_batch_size': 100,
        'size_labeled_data': 4000,
        'train_batch_size': 128,
        'train_batch_size_2': 100,
        'cifar_fname': '',
        'nz': 128,
        'GAN_nz': 128,
        'ngpu': 1,
        'nc':3
    }
    for k, v in optdict.items():
        setattr(opt, k, v)

    if mode == 'vanilla':
        opt.outf = xp+'yogesh_vanillagan_cifar'
        opt.netG = opt.outf+'/netG_iter_489999.pth' # 14.711713683439996 8.034166
        opt.marygan = False
        opt.imageSize = 32
    elif mode == 'marygan':
        opt.outf = xp+'yogesh_marygan_0p2'
        # opt.netG = opt.outf+'/netG_iter_399999.pth'
        opt.netG = opt.outf+'/netG_iter_399999.pth' # 14.770820632929656 8.12888
        opt.marygan = True
        opt.imageSize = 32 
    elif mode == 'acgan':
        opt.outf = xp+'yogesh_acgan_0p2'
        opt.netG = opt.outf+'/netG_iter_449999.pth' # 15.005030857011036 8.024576
        opt.marygan = False
        opt.imageSize = 32

    if opt.netG == '':
        netGfiles = glob.glob(os.path.join(opt.outf, 'netG_iter_*.pth'))
        netGfiles.sort(key = lambda s: int(s.split('_')[-1].split('.')[0]))
        opt.netG = netGfiles[-1]
        print(opt.netG)

    if opt.imageSize == 32:
        netG = resnet.Generator(opt)
    elif opt.imageSize == 64:
        netG = resnet_extra.Generator(opt)
    elif opt.imageSize == 48:
        netG = resnet_48.Generator(opt)
    netG.load_state_dict(torch.load(opt.netG))
    netG = netG.cuda()

    # load comp net
    device = torch.device("cuda:0")
    compnet = DenseNet121()
    compnet = torch.nn.DataParallel(compnet)
    checkpoint = torch.load(os.path.join('/fs/vulcan-scratch/ilyak/locDoc/experiments/classifiers/cifar/densenet121','ckpt_47.t7'))
    compnet.load_state_dict(checkpoint['net'])
    compnet = compnet.to(device)
    compnet.eval();
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    minimal_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # gen images

    batch_size = opt.train_batch_size
    nz = opt.nz
    noise = torch.FloatTensor(opt.train_batch_size, nz)
    noise = noise.cuda()
    num_classes = 10

    # create images
    batch_size = opt.train_batch_size
    nz = opt.nz
    noise = torch.FloatTensor(opt.train_batch_size, nz)
    noise = noise.cuda()
    disc_batch = torch.FloatTensor(opt.train_batch_size, 3, opt.imageSize,opt.imageSize)
    disc_batch = disc_batch.cuda()
    num_classes = 10

    # create images
    n_gen_imgs = ((n_used_imgs // opt.train_batch_size) + 1) * opt.train_batch_size
    x = np.empty((n_gen_imgs,3,opt.imageSize,opt.imageSize), dtype=np.uint8)
    batch_in = np.empty((batch_size,3,opt.imageSize,opt.imageSize))
    # create a bunch of GAN images
    start = 0
    pbar = tqdm(total=n_gen_imgs)
    while not start == n_gen_imgs:
        
    #for l in  tqdm(range((n_used_imgs // opt.train_batch_size) + 1),desc='Generating'):
        noise.data.resize_(batch_size, nz).normal_(0, 1)
        #label = np.random.randint(0, num_classes, batch_size)
        if klass_picked is None:
            label = np.random.randint(0, num_classes, batch_size)
        else:
            label = np.ones((batch_size,),dtype=int)*klass_picked
        noise_ = np.random.normal(0, 1, (batch_size, nz))
        if not opt.marygan:
            class_onehot = np.zeros((batch_size, num_classes))
            class_onehot[np.arange(batch_size), label] = 1
            noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
        noise_ = (torch.from_numpy(noise_))
        noise.data.copy_(noise_.view(batch_size, nz))
        fake = netG(noise).data.cpu().numpy()
        fake = np.floor((fake + 1) * 255/2.0).astype(np.uint8)
        
        if klass_picked is not None:
            for bi in range(fake.shape[0]):
                batch_in[bi] = minimal_trans(np.moveaxis(fake[bi],0,-1))
            disc_batch.data.copy_(torch.from_numpy(batch_in));
            batch_out = compnet(disc_batch).detach().data.cpu()
            class_filt = (np.argmax(batch_out,1) == klass_picked).numpy().astype(bool)
            fake_picked = fake[class_filt]
            end = min(start + fake_picked.shape[0], n_gen_imgs)
            
            x[start:end] = fake_picked[:(end-start)]
            start = end
            pbar.update(fake_picked.shape[0])
        else:
            end = start + batch_size
            x[start:end] = fake
            start = end
            pbar.update(batch_size)
    pbar.close()




    #net_in = np.empty((x.shape[0],3,32,32))
    net_in = np.empty((x.shape[0],3,32,32))
    for i in tqdm(range(x.shape[0]),desc='Preprocess'):
        net_in[i] = minimal_trans(np.moveaxis(x[i],0,-1))

    my_dataset = utils.TensorDataset(torch.FloatTensor(net_in))
    my_dataloader = utils.DataLoader(my_dataset, batch_size=opt.train_batch_size, shuffle=False)



    #net_out = np.empty((x.shape[0], 602112))
    #net_out = np.empty((x.shape[0], 12288))
    net_out = np.empty((x.shape[0], 1024)) # Densenet
    # net_out = np.empty((x.shape[0], 2622)) # vgg-face
    for i, batch in enumerate(tqdm(my_dataloader,desc='Extract Feat')):
        start = i * opt.train_batch_size
        end = start + opt.train_batch_size
        batch_in = batch[0].to(device)
        batch_out = compnet(batch_in, feat=1).detach().data.cpu()
        net_out[start:end] = batch_out


    D = pairwise_distances(net_out)
    # remove the diagonal and lower triangle
    to_del = np.tril(np.ones((D.shape[0], D.shape[0]), dtype=int))
    D[to_del == 1] = D.max()

    dists = D.flatten()
    closest_N = 10
    idxs = np.argpartition(dists,closest_N)
    min_dists = sorted(dists[idxs[:closest_N]])
    min_idxs = sorted(idxs[:closest_N], key=lambda i: dists[i])
    closest_idxs = [(idx // D.shape[0], idx % D.shape[0]) for idx in min_idxs]

    closest_imgs = np.empty((closest_N * 2,)+x.shape[1:])
    for l, (i,j) in enumerate(closest_idxs):
        closest_imgs[2*l] = x[min(i,j)]
        closest_imgs[2*l + 1] = x[max(i,j)]

    fake_grid = vutils.make_grid(torch.Tensor(closest_imgs[:20]), nrow=2, padding=0, normalize=True)
    fake_imgs = np.moveaxis(fake_grid.data.cpu().numpy(),0,-1)
    
    return fake_imgs, min_dists

def run_cifar_trial(mode, klass_picked, trialn=0):
    fake_imgs, min_dists = cifar_plots(mode, klass_picked)

    outfn = '%s_class_%i_birthday_trial_%i.npz' % (mode, klass_picked, trialn)
    
    np.savez(xp+'diversity_plots/cifar/'+outfn,
        imgs=fake_imgs, dists=min_dists)

if __name__ == '__main__':

    for k in range(10):
        for t in range(1,10):
            run_cifar_trial('marygan', k, t)
#             run_cifar_trial('acgan', k, t)
#             run_cifar_trial('vanilla', k, t)

    #     run_stl_trials('marygan', k, 10)
    #     run_stl_trials('acgan', k, 10)
    #     run_stl_trials('vanilla', k, 10)

    

