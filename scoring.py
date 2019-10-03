#!/usr/bin/env python3
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
from torchvision.datasets import CIFAR10, STL10, ImageFolder
import torch
import visdom
from torchvision import transforms
from GAN_training.models import resnet, resnet_extra, resnet_48_flat, resnet_48
from tqdm import tqdm

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
        'train_batch_size_2': 100,
        'dataset_is_fid': '/scratch0/ilya/locDoc/data/cifar10/fid_is_scores.npz'
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
    
    np.savez(optinst.dataset_is_fid, mfid=mfid, sfid=sfid, mis=mis, sis=sis)
    
def calc_stl():
    print('Calculating STL IS and FID')
    N = 50000
    optinst = optclass()
    optdict = {
        'data_root': '/scratch0/ilya/locDoc/data/stl10',
#         'data_root': '/fs/vulcan-scratch/ilyak/locDoc/data/stl10',
        'dataset': 'stl',
        'imageSize': 48,
        'dev_batch_size': 100,
        'size_labeled_data': 4000,
        'train_batch_size': 100,
        'train_batch_size_2': 100,
        'outdata_fname': '/scratch0/ilya/locDoc/data/stl10/fid_is_scores.npz',
#         'outdata_fname': '/fs/vulcan-scratch/ilyak/locDoc/data/stl10/fid_is_scores.npz'
    }
    for k, v in optdict.items():
        setattr(optinst, k, v)

    tform = transforms.Compose([
        transforms.Resize(optinst.imageSize),
        transforms.Lambda(lambda img: np.array(img))])
    
    training_set = STL10(optinst.data_root, split='unlabeled', download=True, transform=tform)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=optinst.train_batch_size_2, shuffle=True, num_workers=2)
    assert(len(trainloader) * optinst.train_batch_size_2 >= N)
    images = np.empty((N,optinst.imageSize,optinst.imageSize,3))
    for i, xy in enumerate(trainloader, 0):
        x, _ = xy
        start = i * optinst.train_batch_size_2
        end = start + optinst.train_batch_size_2
        images[start:end] = np.array(x)
        if end >= N:
            break

    mis, sis = iscore.get_inception_score(images)
    
    print('IS: %f (+/- %f)' % (mis, sis))
    
    mfid, sfid = fid_ms_for_imgs(images)
    
    np.savez(optinst.outdata_fname, mfid=mfid, sfid=sfid, mis=mis, sis=sis)

def calc_celeba():
    print('Calculating celeba FID')
    N = 50000
    optinst = optclass()
    optdict = {
        'data_root': '/scratch0/ilya/locDoc/data/celeba_5class',
        'dataset': 'stl',
        'imageSize': 64,
        'dev_batch_size': 100,
        'size_labeled_data': 4000,
        'train_batch_size': 100,
        'train_batch_size_2': 100,
        'outdata_fname': '/scratch0/ilya/locDoc/data/celeba_5class/fid_is_scores.npz',
    }
    for k, v in optdict.items():
        setattr(optinst, k, v)

    tform = transforms.Compose([
        transforms.Resize((optinst.imageSize,optinst.imageSize)),
        transforms.Lambda(lambda img: np.array(img))])
    
    training_set = ImageFolder(root=optinst.data_root, transform=tform)
    trainloader = torch.utils.data.DataLoader(training_set, batch_size=optinst.train_batch_size_2, shuffle=True, num_workers=2)
    assert(len(trainloader) * optinst.train_batch_size_2 >= N)
    images = np.empty((N,optinst.imageSize,optinst.imageSize,3))
    for i, xy in enumerate(trainloader, 0):
        x, _ = xy
        start = i * optinst.train_batch_size_2
        end = start + optinst.train_batch_size_2
        images[start:end] = np.array(x)
        if end >= N:
            break
    
    mfid, sfid = fid_ms_for_imgs(images)
    
    np.savez(optinst.outdata_fname, mfid=mfid, sfid=sfid)
    
    
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
    data_stats = np.load(opt.dataset_is_fid)
    last_itr = -1
    lf = os.path.join(opt.outf, listen_file)
    slept_last_itr = False
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
            sys.stdout.write('\nWaking\n')
            sys.stdout.flush()
            slept_last_itr = False
            last_itr = now_itr
            path = os.path.join(opt.outf, 'GAN_OUTPUTS', 'out.npy')
            x = np.load(path)
        
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
            if slept_last_itr:
                sys.stdout.write('.')
                sys.stdout.flush()
            else:
                sys.stdout.write('Sleeping')
                sys.stdout.flush()
            time.sleep(5)
            slept_last_itr = True

HIST_FNAME = 'scoring_hist.npy'
def batch_scores(opt):
    num_classes = 10
    print('running batch scores')
    data_stats = np.load(opt.dataset_is_fid)
    scores_todo = opt.scores_todo.split(',')
    
    # make/load history files in each
    def load_or_make_hist(d):
        if not os.path.isdir(d):
            raise Exception('%s is not a valid directory' % d)
        f = os.path.join(d, HIST_FNAME)
        if os.path.isfile(f):
            return np.load(f, allow_pickle=True).item()
        else:
            return defaultdict(dict)
    full_paths = [os.path.join(opt.outf, d) for d in opt.dirs.split(' ')]
    cps = [int(c) for c in opt.checkpoints.split(' ')]
    #hists = [defaultdict(dict) for d in full_paths]
    hists = [load_or_make_hist(d) for d in full_paths]
    legend = [os.path.split(d)[-1] for d in full_paths]
    
    #pdb.set_trace()
    
    if opt.imageSize == 32:
        netG = resnet.Generator(opt)
    elif opt.imageSize == 64:
        netG = resnet_extra.Generator(opt)
    elif opt.imageSize == 48:
        if opt.net_type == 'flat':
            netG = resnet_48_flat.Generator(opt)
        else:
            netG = resnet_48.Generator(opt)
    # check everything exists
    sys.stdout.write('Checking all files exist')
    for i, d in enumerate(full_paths):
        for j, c in enumerate(cps):
            mfG = os.path.join(d, 'netG_iter_%06d.pth' % c)
            if not os.path.isfile(mfG):
                print('File %s NOT FOUND' % mfG)
            else:
                try:
                    netG.load_state_dict(torch.load(mfG))
                except:
                    print('Failed to load %s' % mfG)
                    netG.load_state_dict(torch.load(mfG))
                sys.stdout.write('.')
                sys.stdout.flush()
    
    # setup visdom for monitoring
    visdom_IS_visual_id = None
    visdom_FID_visual_id = None
    vis = visdom.Visdom(env=opt.visdom_board, port=opt.port, server=opt.host)
    
    # loop through files
    display_IS = np.zeros((len(full_paths), len(cps)))
    display_FID = np.zeros((len(full_paths), len(cps)))
    for j, c in enumerate(cps):
        for i, d in enumerate(full_paths):
            mfG = os.path.join(d, 'netG_iter_%06d.pth' % c)
            if ('IS' in hists[i][c]) and ('FID' in hists[i][c]) and not opt.overwrite:
                print('SKIPPING %s, values already computed' % mfG)
                next
            elif os.path.isfile(mfG):
                # tf.reset_default_graph()
                # find and load the model
                if opt.imageSize == 32:
                    netG = resnet.Generator(opt)
                elif opt.imageSize == 64:
                    netG = resnet_extra.Generator(opt)
                elif opt.imageSize == 48:
                    if opt.net_type == 'flat':
                        netG = resnet_48_flat.Generator(opt)
                    else:
                        netG = resnet_48.Generator(opt)
                netG.load_state_dict(torch.load(mfG))
                netG = netG.cuda()

                batch_size = opt.train_batch_size
                nz = opt.nz
                noise = torch.FloatTensor(opt.train_batch_size, nz)
                noise = noise.cuda()

                # create images
                n_used_imgs = 50000
                x = np.empty((n_used_imgs,opt.imageSize,opt.imageSize,3), dtype=np.uint8)
                # create a bunch of GAN images
                for l in  tqdm(range(n_used_imgs // opt.train_batch_size),desc='Generating [%s][%06d]' % (legend[i], c)):
                    start = l * opt.train_batch_size
                    end = start + opt.train_batch_size

                    noise.data.resize_(batch_size, nz).normal_(0, 1)
                    label = np.random.randint(0, num_classes, batch_size)
                    noise_ = np.random.normal(0, 1, (batch_size, nz))
                    
                    if opt.acgan_noise:
                        class_onehot = np.zeros((batch_size, num_classes))
                        class_onehot[np.arange(batch_size), label] = 1
                        noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
                    noise_ = (torch.from_numpy(noise_))
                    noise.data.copy_(noise_.view(batch_size, nz))
                    fake = netG(noise).detach().data.cpu().numpy()

                    fake = np.floor((fake + 1) * 255/2.0).astype(np.uint8)
                    x[start:end] = np.moveaxis(fake,1,-1)
                
                torch.cuda.empty_cache() # without this get the error: Failed to get convolution algorithm
                # scoring starts
                if 'IS' in scores_todo:
                    if (not ('IS' in hists[i][c])) or opt.overwrite: 
                        mis, sis = iscore.get_inception_score(x, mem_fraction=opt.tfmem)
                        print('[%s][%06d] IS mu: %f. IS sigma: %f.' % (legend[i], c, mis, sis))
                        hists[i][c]['IS'] = [mis, sis]
                        np.save(os.path.join(d, HIST_FNAME), hists[i])

                if 'FID' in scores_todo:
                    if (not ('FID' in hists[i][c])) or opt.overwrite: 
                        m1, s1 = fid_ms_for_imgs(x, mem_fraction=opt.tfmem)
                        fid_value = fid.calculate_frechet_distance(m1, s1, data_stats['mfid'], data_stats['sfid'])
                        print('[%s][%06d] FID: %f' % (legend[i], c, fid_value))
                        hists[i][c]['FID'] = fid_value
                        np.save(os.path.join(d, HIST_FNAME), hists[i])
                del x
            else:
                print('SKIPPING %s, no file found' % mfG)
                next
            if ('IS' in hists[i][c]) and (type(hists[i][c]['IS']) is list) and len(hists[i][c]['IS']):
                display_IS[i,j] = hists[i][c]['IS'][0]
            if ('FID' in hists[i][c]):
                display_IS[i,j] = hists[i][c]['FID']

            
            # show in visdom
            visdom_IS_visual_id = vis.line(display_IS.T, cps, win=visdom_IS_visual_id, opts={'legend': legend, 'title': 'IS Scores'})
            visdom_FID_visual_id = vis.line(display_FID.T, cps, win=visdom_FID_visual_id, opts={'legend': legend, 'title': 'FID Scores'})
            
            
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar | imagenet')
    parser.add_argument('--outf', default='.', help='folder to look for output images')
    parser.add_argument('--host', default='http://ramawks69', type=str, help="hostname/server visdom listener is on.")
    parser.add_argument('--port', default=8097, type=int, help="which port visdom should use.")
    parser.add_argument('--visdom_board', default='main', type=str, help="name of visdom board to use.")
    parser.add_argument('--run_scoring_now', type=bool, default=False)
    parser.add_argument('--tfmem', default=0.5, type=float, help="What fraction of GPU memory tf should use.")
    parser.add_argument('--dataset_is_fid', default='/scratch0/ilya/locDoc/data/cifar10/fid_is_scores.npz', help='cifar raw data IS FID stats')
    parser.add_argument('--mode', default='batch', help='listen | batch')
    parser.add_argument('--dirs', help='For batch mode, use space to separate list')
    parser.add_argument('--checkpoints', help='For batch mode, use space to separate list')
    parser.add_argument('--overwrite', type=bool, default=False, help="Include this argument to overwrite batch perf, otherwise omit it")
    parser.add_argument('--net_type', default='flat', help='Only relevant for image size 48')
    parser.add_argument('--scores_todo', default='IS,FID', help="")
    parser.add_argument('--acgan_noise', type=bool, default=False, help="Include this argument to generate noise as input to the generator with class 1 hots.")
    
    # for batch mode from main.py
    parser.add_argument('--train_batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=128)
    parser.add_argument('--ndf', type=int, default=128)
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')

    
    opt = parser.parse_args()
    print(opt)
    
    opt.GAN_nz = opt.nz
    opt.GAN_ngf = opt.ngf
    opt.GAN_ndf = opt.ndf
#     opt.GAN_disc_iters = opt.ndis
#     opt.GAN_beta1 = opt.beta1
    opt.batchSize = opt.train_batch_size
    opt.ngpu = 1
    opt.nc = 3
    
    if opt.dataset == 'cifar' and opt.mode == 'listen':
        cifar_listen(opt)
    elif opt.mode == 'batch':
        batch_scores(opt)
    elif opt.mode == 'calc_stl':
        calc_stl()
    elif opt.mode == 'calc_celeba':
        calc_celeba()
    else:
        raise NotImplementedError("No such dataset {}".format(opt.dataset))

    
    #fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    #print("FID: %s" % fid_value)