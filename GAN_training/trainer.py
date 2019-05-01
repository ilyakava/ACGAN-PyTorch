import torchvision.utils as vutils
import torch.optim as optim
import torch
import torch.nn.functional as F
import GAN_training.utils as utils
import numpy as np


class GAN:
    def __init__(self, netG, netD, dataloader, opt, device):

        self.opt = opt
        self.dataloader = dataloader
        self.netG = netG
        self.netD = netD
        self.optimizerD = optim.Adam(netD.parameters(), lr=opt.GAN_lrD, betas=(opt.GAN_beta1, 0.999))
        self.optimizerG = optim.Adam(netG.parameters(), lr=opt.GAN_lrG, betas=(opt.GAN_beta1, 0.999))

        self.real_label = 1
        self.fake_label = 0
        self.device = device
        self.disc_label = torch.full((opt.batchSize,), self.real_label, device=device)
        self.fixed_noise = utils.sample_normal(opt.batchSize, opt.GAN_nz).to(device)
        self.start_epoch = 0

    def disc_criterion(self, inputs, labels):
        if self.opt.GAN_disc_loss_type == 'wasserstein':
            return torch.mean(inputs*labels) - torch.mean(inputs*(1-labels))
        elif self.opt.GAN_disc_loss_type == 'hinge':
            return torch.mean(F.relu(1 + inputs*labels)) + torch.mean(F.relu(1 - inputs*(1-labels)))
        else:
            return F.binary_cross_entropy(F.sigmoid(inputs), labels)
            # return nn.BCEWithLogitsLoss()(inputs, labels)

    def disc_updates(self, real_data):

        batch_size = real_data.size(0)

        self.disc_label.fill_(self.real_label)
        output_d = self.netD(real_data)
        errD_real = self.disc_criterion(output_d, self.disc_label)
        errD_real.backward(retain_graph=True)

        # train with fake
        noise = utils.sample_normal(batch_size, self.opt.GAN_nz).to(self.device)
        fake = self.netG(noise)
        self.disc_label.fill_(self.fake_label)
        output_d = self.netD(fake.detach())
        errD_fake = self.disc_criterion(output_d, self.disc_label)
        errD_fake.backward()
        self.optimizerD.step()

        disc_loss = errD_real + errD_fake
        return disc_loss.item()

    def gen_updates(self, real_data):

        batch_size = real_data.size(0)

        self.netG.zero_grad()
        noise = utils.sample_normal(batch_size, self.opt.GAN_nz).to(self.device)
        fake = self.netG(noise)
        self.disc_label.fill_(self.real_label)
        output_d = self.netD(fake)
        errG_disc = self.disc_criterion(output_d, self.disc_label)
        errG_disc.backward()
        self.optimizerG.step()

        return errG_disc.item()

    def train(self):

        for epoch in range(self.start_epoch, self.opt.GAN_nepochs):
            for i, data in enumerate(self.dataloader, 0):

                # Forming data and label tensors
                self.netD.zero_grad()
                real_data = data[0].to(self.device)

                # Updates
                real_disc_loss = self.disc_updates(real_data)

                if i % self.opt.GAN_disc_iters == 0:
                    fake_disc_loss = self.gen_updates(real_data)

                if i % 20 == 0:
                    print(
                        '[{}/{}][{}/{}] Real disc loss: {}, Fake disc loss: '
                        '{}'.format(epoch, self.opt.nepochs, i, len(self.dataloader),
                                    real_disc_loss, fake_disc_loss))

                if i % 100 == 0:
                    vutils.save_image(real_data * 0.5 + 0.5,
                                      '%s/real_samples.png' % self.opt.GAN_outf,
                                      normalize=False)
                    fake = self.netG(self.fixed_noise)
                    vutils.save_image((fake.detach()) * 0.5 + 0.5,
                                      '%s/fake_samples_epoch_%03d.png' % (self.opt.GAN_outf, epoch),
                                      normalize=False)

            # do checkpointing
            disc_state = {
                'epoch': epoch,
                'state_dict': self.netD.state_dict(),
                'optimizer_state_dict': self.optimizerD.state_dict()
            }
            gen_state = {
                'epoch': epoch,
                'state_dict': self.netG.state_dict(),
                'optimizer_state_dict': self.optimizerG.state_dict()
            }
            torch.save(disc_state, '{}/netD_{}.pth'.format(self.opt.GAN_outf, int(epoch / 5)))
            torch.save(gen_state, '{}/netG_{}.pth'.format(self.opt.GAN_outf, int(epoch / 5)))


class Encoder:
    def __init__(self, netG, netE, dataloader, opt, device):

        self.opt = opt
        self.dataloader = dataloader
        self.netG = netG
        self.netE = netE
        self.optimizerE = optim.Adam(netE.parameters(), lr=opt.enc_lr, betas=(opt.enc_beta1, 0.999))
        self.device = device
        self.start_epoch = 0
        self.latent_const = 0.1
        self.latent_thresh = np.sqrt(self.opt.GAN_nz) + 2

    def enc_updates(self, real_data, use_latent_loss=True):

        self.netE.zero_grad()
        enc_code = self.netE(real_data)
        dec_samples = self.netG(enc_code)

        loss = F.mse_loss(dec_samples, real_data)

        if use_latent_loss:
            enc_code_dec = self.netE(dec_samples)
            # loss_latent = F.mse_loss(enc_code, enc_code_dec)
            loss_latent = torch.sum(F.relu(torch.norm(enc_code, dim=1) - self.latent_thresh))
            loss += self.latent_const*loss_latent

        loss.backward()
        self.optimizerE.step()

        return loss.item(), enc_code

    def train(self):

        for epoch in range(self.start_epoch, self.opt.enc_nepochs):
            for i, (data_normal, data_ano) in enumerate(self.dataloader, 0):

                real_data = data_normal[0].to(self.device)

                # Updates
                enc_loss, enc_code = self.enc_updates(real_data, use_latent_loss=True)
                enc_code_norm = torch.norm(enc_code, dim=1)
                enc_code_norm_mean = torch.mean(enc_code_norm)

                if i % 20 == 0:
                    print(
                        '[{}/{}][{}/{}] Encoder loss: {}, Encoder norm mean: {}'.format(epoch, self.opt.nepochs, i,
                                                                                        len(self.dataloader), enc_loss,
                                                                                        enc_code_norm_mean))

                if i % 100 == 0:
                    vutils.save_image(real_data * 0.5 + 0.5,
                                      '%s/real_samples.png' % self.opt.enc_outf,
                                      normalize=False)
                    fake = self.netG(self.netE(real_data))
                    vutils.save_image((fake.detach()) * 0.5 + 0.5,
                                      '%s/fake_samples_epoch_%03d.png' % (self.opt.enc_outf, epoch),
                                      normalize=False)

            # do checkpointing
            enc_state = {
                'epoch': epoch,
                'state_dict': self.netE.state_dict(),
                'optimizer_state_dict': self.optimizerE.state_dict()
            }
            torch.save(enc_state, '{}/netE_{}.pth'.format(self.opt.enc_outf, int(epoch / 5)))

