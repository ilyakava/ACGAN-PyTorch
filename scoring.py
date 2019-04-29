#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import argparse
import os
import glob
import time
import pathlib
import imageio
import numpy as np
import fid
import inception as iscore
import imageio
import tensorflow as tf
from torchvision.datasets import CIFAR10
import torch
import visdom
from torchvision import transforms

import data

import pdb

CIFAR_FNAME = '/scratch0/ilya/locDoc/data/cifar10/fid_is_scores.npz'

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

    mis, sis = iscore.get_inception_score(images)
    
    print('IS: %f (+/- %f)' % (mis, sis))
    
    mfid, sfid = fid_ms_for_imgs(images)
    
    np.savez(CIFAR_FNAME, mfid=mfid, sfid=sfid, mis=mis, sis=sis)
    
    
    
def fid_ms_for_imgs(images, mem_fraction=1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_fraction)
    inception_path = fid.check_or_download_inception(None)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)
    return mu_gen, sigma_gen
    
    
def cifar_listen(opt, listen_file='scoring.info', write_file='scoring.npy'):
    visdom_score_visual_id = None
    vis = visdom.Visdom(env=opt.visdom_board, port=opt.port, server=opt.host)
    data_stats = np.load(CIFAR_FNAME)
    last_itr = -1
    lf = os.path.join(opt.outf, listen_file)
    if not opt.run_scoring_now:
        if os.path.isfile(lf):
            f = open(lf,'r')
            last_itr = f.read()
            f.close()
    
    history = []
    if opt.run_scoring_now:
        print('Running scoring now for: %s...' % lf)
    else:
        print('Beginning to listen for: %s...' % lf)
        
    while True:
        now_itr = last_itr # init
        if os.path.isfile(lf): # set now_itr
            f = open(lf,'r')
            now_itr = int(f.read())
            f.close()
        
        if now_itr != last_itr:
            last_itr = now_itr
            path = os.path.join(opt.outf, 'GAN_OUTPUTS')
            path = pathlib.Path(path)
            files = list(path.glob('*.png'))
            x = np.array([imageio.imread(str(fn)).astype(np.float32) for fn in files])
        
            mis, sis = iscore.get_inception_score(x, mem_fraction=opt.tfmem)
            print('[%06d] IS mu: %f. IS sigma: %f.' % (now_itr, mis, sis))
            
            m1, s1 = fid_ms_for_imgs(x, mem_fraction=opt.tfmem)
            fid_value = fid.calculate_frechet_distance(m1, s1, data_stats['mfid'], data_stats['sfid'])
            print('[%06d] FID: %f' % (now_itr, fid_value))
            
            # display            
            history.append([now_itr, mis, sis, fid_value])
            nphistory = np.array(history)
            
            visdom_score_visual_id = vis.line(nphistory[:,1:], nphistory[:,0], win=visdom_score_visual_id, opts={'legend': ['IS mu', 'IS sigma', 'FID'], 'title': 'Scores'})
            np.save(os.path.join(opt.outf, write_file), nphistory)
                
            del x #clean up memory
        else:
            time.sleep(5)
                
            
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar | imagenet')
    parser.add_argument('--outf', default='.', help='folder to look for output images')
    parser.add_argument('--host', default='http://ramawks69', type=str, help="hostname/server visdom listener is on.")
    parser.add_argument('--port', default=8097, type=int, help="which port visdom should use.")
    parser.add_argument('--visdom_board', default='main', type=str, help="name of visdom board to use.")
    parser.add_argument('--run_scoring_now', type=bool, default=False)
    parser.add_argument('--tfmem', default=0.5, type=float, help="What fraction of GPU memory tf should use.")
    
    opt = parser.parse_args()
    print(opt)
    
    if opt.dataset == 'cifar':
        cifar_listen(opt)
    else:
        raise NotImplementedError("No such dataset {}".format(opt.dataset))

    
    #fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    #print("FID: %s" % fid_value)