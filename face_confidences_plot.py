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
import imageio
import torch
import torchvision.utils as vutils
import torch.utils.data as utils
import visdom
from torchvision import transforms
from GAN_training.models import resnet, resnet_extra, resnet_48
from classification.models.vgg_face_dag import vgg_face_dag
from tqdm import tqdm

import data

from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

from mtcnn.mtcnn import MTCNN
import cv2

import pdb

xp = '/scratch0/ilya/locDoc/ACGAN/experiments/'
# xp = '/fs/vulcan-scratch/ilyak/locDoc/experiments/'
fig_dir = '~/ilyakavalerov@gmail.com/ramawks69/ACGAN-PyTorch/figs/'


def face_plots(mode, n_used_imgs):
    class optclass:
        workaround = True
    
        
    opt = optclass()
    optdict = {
        #'outf': '/scratch0/ilya/locDoc/ACGAN/experiments/yogesh_acgan_0p2',
        'outf': xp+'marygan-stl-48-miyato-hyp-lrGp4-auxp4',
        'netG': xp+'marygan-stl-48-miyato-hyp-lrGp4-auxp4/netG_iter_069999.pth',
        'marygan': True,
        'imageSize': 48,
        'data_root': '/scratch0/ilya/locDoc/data/stl10',
        'dataset': 'cifar',
        'dev_batch_size': 100,
        'size_labeled_data': 4000,
        'train_batch_size': 128,
        'train_batch_size_2': 100,
        'cifar_fname': '/scratch0/ilya/locDoc/data/cifar10/fid_is_scores.npz',
        'nz': 128,
        'GAN_nz': 128,
        'ngpu': 1,
        'nc':3
    }
    for k, v in optdict.items():
        setattr(opt, k, v)

    if mode == 'vanilla':
        opt.outf = xp+'celeb_cpy/celeba_vanillagan'
        opt.netG = opt.outf+'/netG_iter_129999.pth' # 4.130121811844333
        opt.marygan = False
        opt.imageSize = 64
    elif mode == 'marygan':
        opt.outf = xp+'celeb_cpy/celeba5c_marygan'
        opt.netG = opt.outf+'/netG_iter_129999.pth' # 3.6509132644800673
        opt.marygan = True
        opt.imageSize = 64
    elif mode == 'acgan':
        opt.outf = xp+'celeb_cpy/celeba5c_acgan'
        opt.netG = opt.outf+'/netG_iter_119999.pth' # 5.074366134380284
        opt.marygan = False
        opt.imageSize = 64

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

    detector = MTCNN()

    # gen images

    batch_size = opt.train_batch_size
    nz = opt.nz
    noise = torch.FloatTensor(opt.train_batch_size, nz)
    noise = noise.cuda()
    num_classes = 10
    klass_picked = None

    # create images
    n_gen_imgs = ((n_used_imgs // opt.train_batch_size) + 1) * opt.train_batch_size
    confs = np.zeros(n_gen_imgs)
    # create a bunch of GAN images
    for l in tqdm(range((n_used_imgs // opt.train_batch_size) + 1),desc='Generating'):
        start = l * opt.train_batch_size
        end = start + opt.train_batch_size
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

        for didx in range(fake.shape[0]):
            img = np.pad(np.moveaxis(fake[didx],0,-1), ((6,6),(6,6),(0,0)), 'edge')
            res = detector.detect_faces(img)
            if res:
                confs[start + didx] = res[0]['confidence']

    return confs

if __name__ == '__main__':

    ngen = 10001
    for trial in range(10):
        for mode in ['vanilla', 'marygan', 'acgan']:
            outfn = '%s_genW_%i_trial_%i.npz' % (mode, ngen, trial)
            np.savez(xp+'confidence_plots/'+outfn, confs=face_plots(mode, ngen))




    

