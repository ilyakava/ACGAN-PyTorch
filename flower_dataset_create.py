import os
from shutil import copyfile
import glob

import scipy.io as sio
import pdb

outp = '/scratch0/ilya/locDoc/data/flower_102'


imgp = '/scratch0/ilya/locDoc/data/oxford-flowers/102'
imgfiles = glob.glob(os.path.join(imgp, '*.jpg'))
mat_contents = sio.loadmat('/scratch0/ilya/locDownloads/imagelabels.mat')
labels = mat_contents['labels'][0]

imgnames = [imgfile.split('/')[-1] for imgfile in imgfiles]
imgnames.sort()

assert(len(labels) == len(imgnames))
dataset = zip(imgnames, labels)

for imgname, label in dataset:
    class_dir = os.path.join(outp, str(label))
    if not os.path.isdir(class_dir):
        os.makedirs(class_dir)
    src = os.path.join(imgp, imgname)
    dst = os.path.join(class_dir, imgname)
    copyfile(src, dst)

print('done')
