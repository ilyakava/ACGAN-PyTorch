#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
import numpy as np
import fid
import inception_score as iscore
import imageio
import tensorflow as tf
from torchvision.datasets import CIFAR10
import torch
from torchvision import transforms

import data

import pdb

class optclass:
    workaround = True

def calc_cifar():
    N = 50000
    optinst = optclass()
    optdict = {
        'data_root': '/scratch0/ilya/locDoc/data/cifar10',
        'dataset': 'cifar',
        'dev_batch_size': 100,
        'size_labeled_data': 4000,
        'train_batch_size': 100,
        'train_batch_size_2': 100
    }
    for k, v in optdict.items():
        setattr(optinst, k, v)

    
    training_set = CIFAR10(optinst.data_root, train=True, download=True, transform=transforms.Lambda(lambda img: np.array(img)))
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=optinst.train_batch_size_2, shuffle=True, num_workers=2)
    assert(len(trainloader) * optinst.train_batch_size_2 == N)
    images = np.empty((N,32,32,3))
    for i, xy in enumerate(trainloader, 0):
        x, _ = xy
        start = i * optinst.train_batch_size_2
        end = start + optinst.train_batch_size_2
        images[start:end] = np.array(x)

    mfid, sfid = fid_ms_for_imgs(images)
    
    mis, sis = iscore.get_inception_score(images)
    
    print('IS: %f (+/- %f)' % (mis, sis))
    
    np.savez('/scratch0/ilya/locDoc/data/cifar10/fid_is_scores.npz', mfid=mfid, sfid=sfid, mis=mis, sis=sis)
    
    
    
def fid_ms_for_imgs(images):
    inception_path = fid.check_or_download_inception(None)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)
    return mu_gen, sigma_gen
    

if __name__ == '__main__':
    calc_cifar()
    
    #fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    #print("FID: %s" % fid_value)