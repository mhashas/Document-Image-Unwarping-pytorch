import torch
from tqdm import tqdm

from core.trainers.trainer import Trainer
from util.general_functions import get_discriminator_model, get_optimizer, set_requires_grad, get_loss_function
from constants import *

class GANTrainer(Trainer):

    def __init__(self, args):
        super(GANTrainer, self).__init__(args)

        self.model_D = get_discriminator_model(args)
        self.optimizer_D = get_optimizer(self.model_D, args)
        self.criterion_D = get_loss_function(args.gan_loss_type)

    def run_epoch(self, epoch, split=TRAIN):
        loss = 0.0

        if split == TRAIN:
            self.model.train()
            loader = self.train_loader
        elif split == VAL:
            self.model.eval()
            loader = self.val_loader
        else:
            self.model.eval()
            loader = self.test_loader

        bar = tqdm(loader)
        num_img = len(loader)

        for i, sample in enumerate(bar):
            with torch.autograd.set_detect_anomaly(True):
                image = sample[0]
                target = sample[1]

                if self.args.cuda:
                    image, target = image.cuda(), target.cuda()

                self.scheduler(self.optimizer, i, epoch, self.best_loss)

                if split == TRAIN:
                    output = self.model(image)
                else:
                    with torch.no_grad():
                        output = self.model(image)

                set_requires_grad(self.model_D, True)  # enable backprop for D
                self.optimizer_D.zero_grad()
                loss_D = self.calculate_gan_loss_D(image, output, target)

                if split == TRAIN:
                    loss_D.backward()
                    self.optimizer_D.step()

                # update G
                set_requires_grad(self.model_D, False)  # D requires no gradients when optimizing G
                self.optimizer.zero_grad()
                loss_G = self.calculate_gan_loss_G(image, output, target)

                # Show 10 * 3 inference results each epoch
                if i % (num_img // 10) == 0:
                    self.summary.visualize_image(image, target, output, split=split)

                if split == TRAIN:
                    loss_G.backward()

                    if self.args.clip > 0:
                        if self.args.gpu_ids:
                            torch.nn.utils.clip_grad_norm_(self.model.module().parameters(), self.args.clip)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

                    self.optimizer.step()

                loss += loss_G.item()
                bar.set_description(split +' loss: %.3f' % (loss / (i + 1)))

        if loss < self.best_loss:
            self.best_loss = loss

        self.summary.add_scalar(split + '/total_loss_epoch', loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))

    def calculate_gan_loss_D(self, image, output, target):
        fake = torch.cat((image, output), 1)
        pred_fake = self.model_D(fake.detach())

        real = torch.cat((image, target), 1)
        pred_real = self.model_D(real)

        loss_D_real = self.criterion_D(pred_real, True)
        loss_D_fake = self.criterion_D(pred_fake, False)
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        return loss_D

    def calculate_gan_loss_G(self, image , output, target):
        loss_G = self.criterion(output, target)

        fake = torch.cat((image, output), 1)
        pred_fake = self.model_D(fake)
        loss_G_GAN = self.criterion_D(pred_fake, True)

        loss_G += loss_G_GAN

        return loss_G
