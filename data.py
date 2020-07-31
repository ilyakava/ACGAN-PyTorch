
import os
import sys
import pickle

import numpy as np
import torch
from torchvision.datasets import MNIST, SVHN, CIFAR10, STL10, ImageFolder, CIFAR100
from torchvision import transforms
import torchvision.utils as vutils
from imagenet import ImageNet

import pdb

class CIFAR20(CIFAR100):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # now load the picked numpy arrays
        if self.train:
            self.train_data = []
            self.train_labels = []
            for fentry in self.train_list:
                f = fentry[0]
                file = os.path.join(self.root, self.base_folder, f)
                fo = open(file, 'rb')
                if sys.version_info[0] == 2:
                    entry = pickle.load(fo)
                else:
                    entry = pickle.load(fo, encoding='latin1')
                self.train_data.append(entry['data'])
                if 'labels' in entry:
                    self.train_labels += entry['labels']
                else:
                    self.train_labels += entry['coarse_labels']
                fo.close()

            self.train_data = np.concatenate(self.train_data)
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            f = self.test_list[0][0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.test_data = entry['data']
            if 'labels' in entry:
                self.test_labels = entry['labels']
            else:
                self.test_labels = entry['coarse_labels']
            fo.close()
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC


class DataLoader(object):

    def __init__(self, config, raw_loader, indices, batch_size):
        self.images, self.labels = [], []
        for idx in indices:
            image, label = raw_loader[idx]
            self.images.append(image)
            self.labels.append(label)

        self.images = torch.stack(self.images, 0)
        self.labels = torch.from_numpy(np.array(self.labels, dtype=np.int64)).squeeze()

        if config.dataset == 'mnist':
            self.images = self.images.view(self.images.size(0), -1)

        self.batch_size = batch_size

        self.unlimit_gen = self.generator(True)
        self.len = len(indices)

    def get_zca_cuda(self, reg=1e-6):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        mean = images.mean(0)
        images -= mean.expand_as(images)
        sigma = torch.mm(images.transpose(0, 1), images) / images.size(0)
        U, S, V = torch.svd(sigma)
        components = torch.mm(torch.mm(U, torch.diag(1.0 / torch.sqrt(S) + reg)), U.transpose(0, 1))
        return components, mean

    def apply_zca_cuda(self, components):
        images = self.images.cuda()
        if images.dim() > 2:
            images = images.view(images.size(0), -1)
        self.images = torch.mm(images, components.transpose(0, 1)).cpu()

    def generator(self, inf=False):
        while True:
            indices = np.arange(self.images.size(0))
            np.random.shuffle(indices)
            indices = torch.from_numpy(indices)
            for start in range(0, indices.size(0) - self.batch_size, self.batch_size):
                end = min(start + self.batch_size, indices.size(0))
                ret_images, ret_labels = self.images[indices[start: end]], self.labels[indices[start: end]]
                yield ret_images, ret_labels
            if not inf: break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return self.len

def get_mnist_loaders(config):
    transform = transforms.Compose([transforms.ToTensor()])
    training_set = MNIST(config.data_root, train=True, download=True, transform=transform)
    dev_set = MNIST(config.data_root, train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: int(config.size_labeled_data / 10)]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print ('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0])

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set
    
def get_mnist_subset_loaders(config):
    # TODO pay attention to config.size_labeled_data
    whitelist_labels = [0,1]
    # whitelist_labels = list(range(10))
    # TODO add rotation etc.
    # normalizing MNIST with transforms.Normalize((0.1307,), (0.3081,)), leads
    # to early collapse
    transform=transforms.Compose([
                        
                        transforms.Resize((config.imageSize,config.imageSize)),
                        transforms.ToTensor(),
                       ])
                        # transforms.Lambda(lambda x: x.repeat(3, 1, 1) ),
    training_set = MNIST(config.data_root, train=True, download=True, transform=transform)
    dev_set = MNIST(config.data_root, train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    # np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    
    for i in whitelist_labels:
        mask[np.where(labels == i)[0]] = True
    # relabel based on whitelist
    # for new_label, old_label in enumerate(whitelist_labels):
    #     for idx in np.where(labels == old_label)[0]:
    #         pdb.set_trace()
    #         training_set[idx][1] = new_label # does not support assignment
            
        
        
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    print ('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0])

    np.random.shuffle(labeled_indices)
    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    # unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    # unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    # dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in whitelist_labels:
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, None, None, None, special_set

def get_svhn_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = SVHN(config.data_root, split='train', download=True, transform=transform)
    dev_set = SVHN(config.data_root, split='test', download=True, transform=transform)

    def preprocess(data_set):
        for i in range(len(data_set.data)):
            if data_set.labels[i] == 10:
                data_set.labels[i] = 0
    preprocess(training_set)
    preprocess(dev_set)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][:(config.size_labeled_data // 10)]] = True
    # labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    labeled_indices, unlabeled_indices = indices[mask], indices
    print ('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(dev_set))

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size)
    unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set

def get_cifar_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = CIFAR10(config.data_root, train=True, download=True, transform=transform)
    dev_set = CIFAR10(config.data_root, train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    for i in range(10):
        mask[np.where(labels == i)[0][: int(config.size_labeled_data / 10)]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    # labeled_indices, unlabeled_indices = indices[mask], indices
    print ('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(dev_set))
    unlabeled_loader = None
    unlabeled_loader2 = None

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    if unlabeled_indices.shape[0]:
        unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
        unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set

def get_cifar100_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = CIFAR100(config.data_root, train=True, download=True, transform=transform)
    dev_set = CIFAR100(config.data_root, train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    nclass = 100
    for i in range(nclass):
        mask[np.where(labels == i)[0][: int(config.size_labeled_data / nclass)]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    # labeled_indices, unlabeled_indices = indices[mask], indices
    print ('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(dev_set))
    unlabeled_loader = None
    unlabeled_loader2 = None

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    if unlabeled_indices.shape[0]:
        unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
        unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set

def get_cifar20_loaders(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = CIFAR20(config.data_root, train=True, download=True, transform=transform)
    dev_set = CIFAR20(config.data_root, train=False, download=True, transform=transform)

    indices = np.arange(len(training_set))
    np.random.shuffle(indices)
    mask = np.zeros(indices.shape[0], dtype=np.bool)
    labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    nclass = 20
    for i in range(nclass):
        mask[np.where(labels == i)[0][: int(config.size_labeled_data / nclass)]] = True
    labeled_indices, unlabeled_indices = indices[mask], indices[~ mask]
    # labeled_indices, unlabeled_indices = indices[mask], indices
    print ('labeled size', labeled_indices.shape[0], 'unlabeled size', unlabeled_indices.shape[0], 'dev size', len(dev_set))
    unlabeled_loader = None
    unlabeled_loader2 = None

    labeled_loader = DataLoader(config, training_set, labeled_indices, config.train_batch_size)
    if unlabeled_indices.shape[0]:
        unlabeled_loader = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
        unlabeled_loader2 = DataLoader(config, training_set, unlabeled_indices, config.train_batch_size_2)
    dev_loader = DataLoader(config, dev_set, np.arange(len(dev_set)), config.dev_batch_size)

    special_set = []
    for i in range(10):
        special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set


class MyLiteDataLoader(object):

    def __init__(self, raw_loader, batch_size):
        self.unlimit_gen = self.generator(True)
        self.raw_loader = raw_loader
        self.bs = batch_size
    
    def generator(self, inf=False):
        while True:
            theloader = torch.utils.data.DataLoader(self.raw_loader, batch_size=self.bs, shuffle=True, num_workers=2,drop_last=True)
            for xy in theloader:
                x, y = xy
                yield x, y
            if not inf: break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return len(self.raw_loader)

class MyLiteMergedDataLoader(object):

    def __init__(self, raw_loader1, raw_loader2, batch_size):
        self.unlimit_gen = self.generator(True)
        self.raw_loader1 = raw_loader1
        self.raw_loader2 = raw_loader2
        self.bs = batch_size
    
    def generator(self, inf=False):
        while True:
            for raw_loader in [self.raw_loader1, self.raw_loader2]:
                theloader = torch.utils.data.DataLoader(raw_loader, batch_size=self.bs, shuffle=True, num_workers=2,drop_last=True)
                for xy in theloader:
                    x, y = xy
                    yield x, y
            if not inf: break

    def next(self):
        return next(self.unlimit_gen)

    def get_iter(self):
        return self.generator()

    def __len__(self):
        return len(self.raw_loader1) + len(self.raw_loader2)

def get_stl_loaders(config):
    transform = transforms.Compose([transforms.Resize(config.imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    training_set = STL10(config.data_root, split='train', download=True, transform=transform)
    dev_set = STL10(config.data_root, split='test', download=True, transform=transform)
    unl_set = STL10(config.data_root, split='unlabeled', download=True, transform=transform)

    # print ('labeled size', len(training_set), 'unlabeled size', len(unl_set), 'dev size', len(dev_set))

    # indices = np.arange(len(training_set))
    # labeled_loader = MyLiteDataLoader(training_set, config.train_batch_size)
    labeled_loader = MyLiteMergedDataLoader(training_set, dev_set, config.train_batch_size)
    unlabeled_loader = MyLiteDataLoader(unl_set, config.train_batch_size_2)
    unlabeled_loader2 = None #DataLoader(config, unl_set, np.arange(len(unl_set)), config.train_batch_size_2)
    dev_loader = None
    # dev_loader = MyLiteDataLoader(dev_set, config.dev_batch_size)

    # labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    special_set = []
    #for i in range(10):
    #    special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    #special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set

def get_imagenet_loaders(config):
    transform=transforms.Compose([
                        transforms.Resize((config.imageSize,config.imageSize)),
                         #  transforms.CenterCrop(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ])
    training_set = ImageNet(config.data_root, split='train', download=True, transform=transform)
    dev_set = ImageNet(config.data_root, split='val', download=True, transform=transform)

    print ('labeled size', len(training_set), 'unlabeled size', 0, 'dev size', len(dev_set))

    # indices = np.arange(len(training_set))
    labeled_loader = MyLiteDataLoader(training_set, config.train_batch_size)
    unlabeled_loader = None # MyLiteDataLoader(unl_set, config.train_batch_size_2)
    unlabeled_loader2 = None # DataLoader(config, unl_set, np.arange(len(unl_set)), config.train_batch_size_2)
    dev_loader = MyLiteDataLoader(dev_set, config.dev_batch_size)

    # labels = np.array([training_set[i][1] for i in indices], dtype=np.int64)
    special_set = []
    #for i in range(10):
    #    special_set.append(training_set[indices[np.where(labels==i)[0][0]]][0])
    #special_set = torch.stack(special_set)

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set


def get_celeba_loaders(config):
    transform=transforms.Compose([
                        transforms.Resize((config.imageSize,config.imageSize)),
                         #  transforms.CenterCrop(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ])
    training_set = ImageFolder(root=config.data_root, transform=transform)
    print ('labeled size', len(training_set), 'unlabeled size', 0, 'dev size', 0)

    labeled_loader = MyLiteDataLoader(training_set, config.train_batch_size)
    unlabeled_loader = None
    unlabeled_loader2 = None
    dev_loader = None
    special_set = []

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set

def get_flower102_loaders(config):
    transform=transforms.Compose([
                        transforms.Resize((config.imageSize,config.imageSize)),
                         #  transforms.CenterCrop(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                       ])
    training_set = ImageFolder(root=config.data_root, transform=transform)
    print ('labeled size', len(training_set), 'unlabeled size', 0, 'dev size', 0)

    labeled_loader = MyLiteDataLoader(training_set, config.train_batch_size)
    unlabeled_loader = None
    unlabeled_loader2 = None
    dev_loader = None
    special_set = []

    return labeled_loader, unlabeled_loader, unlabeled_loader2, dev_loader, special_set
