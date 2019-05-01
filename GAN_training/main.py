from __future__ import print_function
import torch.utils.data
from GAN_training import utils
from GAN_training import trainer
from GAN_training.models import DCGAN, DCGAN_spectralnorm, resnet


def create_models(opt):

    if opt.GAN_model == 'dcgan':

        netG = DCGAN.Generator(opt)
        netG.apply(utils.weights_init)
        print(netG)

        netD = DCGAN.Discriminator(opt)
        netD.apply(utils.weights_init)
        print(netD)

    elif opt.GAN_model == 'dcgan_spectral':

        netG = DCGAN_spectralnorm.Generator(opt)
        netG.apply(utils.weights_init)
        print(netG)

        netD = DCGAN_spectralnorm.Discriminator(opt)
        netD.apply(utils.weights_init_spectral)
        print(netD)

    elif opt.GAN_model == 'resnet':

        netG = resnet.Generator(opt)
        print(netG)

        netD = resnet.Discriminator(opt)
        print(netD)

    else:
        raise ValueError('Invalid method specified')

    return netG, netD


def create_encoder(opt):

    if opt.GAN_model == 'dcgan':

        netE = DCGAN.Encoder(opt)
        netE.apply(utils.weights_init)
        print(netE)

    elif opt.GAN_model == 'dcgan_spectral':

        netE = DCGAN_spectralnorm.Encoder(opt)
        netE.apply(utils.weights_init)
        print(netE)

    elif opt.GAN_model == 'resnet':

        netE = resnet.Encoder(opt)
        print(netE)

    else:
        raise ValueError('Invalid method specified')

    return netE


def main_GAN(opt, dataloader):

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    # Creating models
    netG, netD = create_models(opt)
    netG = netG.to(device)
    netD = netD.to(device)

    # Train the model
    trainer_ = trainer.GAN(netG, netD, dataloader, opt, device)

    if opt.GAN_Gpath != '':
        print('Loading generator ...')
        G_state = torch.load(opt.GAN_Gpath)
        trainer_.netG.load_state_dict(G_state['state_dict'])
        trainer_.optimizerG.load_state_dict(G_state['optimizer_state_dict'])
        trainer_.start_epoch = G_state['epoch']

    if opt.GAN_Dpath != '':
        print('Loading discriminator ...')
        D_state = torch.load(opt.GAN_Dpath)
        trainer_.netD.load_state_dict(D_state['state_dict'])
        trainer_.optimizerD.load_state_dict(D_state['optimizer_state_dict'])

    if opt.GAN_istrain:
        print('Training GAN from epoch {}'.format(trainer_.start_epoch))
        trainer_.train()

    return netG, netD


def main_encoder(opt, dataloader, netG):

    device = torch.device("cuda:0" if opt.cuda else "cpu")

    # Creating models
    netE = create_encoder(opt)
    netE.to(device)

    # Train the model
    trainer_ = trainer.Encoder(netG, netE, dataloader, opt, device)

    if opt.enc_path != '':
        print('Loading encoder ...')
        E_state = torch.load(opt.enc_path)
        trainer_.netE.load_state_dict(E_state['state_dict'])
        trainer_.optimizerE.load_state_dict(E_state['optimizer_state_dict'])
        trainer_.start_epoch = E_state['epoch']

    if opt.enc_istrain:
        print('Training encoder from epoch {}'.format(trainer_.start_epoch))
        trainer_.train()

    return netE
