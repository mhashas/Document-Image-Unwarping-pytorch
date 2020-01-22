import torch
import torch.nn as nn
import torch.nn.functional as F

from util.ssim import MS_SSIM, SSIM
from util.ms_ssim import MS_SSIM_v2, SSIM_v2

class LS_GAN_Loss(nn.Module):

    def __init__(self, target_fake_label=0.0, target_real_label=1.0):
        super(LS_GAN_Loss, self).__init__()
        self.loss = nn.MSELoss()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        target_tensor = target_tensor.expand_as(prediction)

        return target_tensor.cuda() if torch.cuda.is_available() else target_tensor

    def forward(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)

        return loss

class DocunetLoss_v2(nn.Module):
    def __init__(self, r=0.1,reduction='mean'):
        super(DocunetLoss_v2, self).__init__()
        assert reduction in ['mean','sum'], " reduction must in ['mean','sum']"
        self.r = r
        self.reduction = reduction

    def forward(self, y, label):
        bs, n, h, w = y.size()
        d = y - label
        loss1 = []
        for d_i in d:
            loss1.append(torch.abs(d_i).mean() - self.r * torch.abs(d_i.mean()))
        loss1 = torch.stack(loss1)
        # lossb1 = torch.max(y1, torch.zeros(y1.shape).to(y1.device)).mean()
        loss2 = F.mse_loss(y, label,reduction=self.reduction)

        if self.reduction == 'mean':
            loss1 = loss1.mean()
        elif self.reduction == 'sum':
            loss1= loss1.sum()
        return loss1 + loss2

class DocunetLoss(nn.Module):
    def __init__(self, lamda=0.1, reduction='mean'):
        super(DocunetLoss, self).__init__()
        self.lamda = lamda
        self.reduction = reduction

    def forward(self, output, target):
        x = target[:, 0, :, :]
        y = target[:, 1, :, :]
        back_sign_x, back_sign_y = (x == -1).int(), (y == -1).int()
        # assert back_sign_x == back_sign_y

        back_sign = ((back_sign_x + back_sign_y) == 2).float()
        fore_sign = 1 - back_sign

        loss_term_1_x = torch.sum(torch.abs(output[:, 0, :, :] - x) * fore_sign) / torch.sum(fore_sign)
        loss_term_1_y = torch.sum(torch.abs(output[:, 1, :, :] - y) * fore_sign) / torch.sum(fore_sign)
        loss_term_1 = loss_term_1_x + loss_term_1_y

        loss_term_2_x = torch.abs(torch.sum((output[:, 0, :, :] - x) * fore_sign)) / torch.sum(fore_sign)
        loss_term_2_y = torch.abs(torch.sum((output[:, 1, :, :] - y) * fore_sign)) / torch.sum(fore_sign)
        loss_term_2 = loss_term_2_x + loss_term_2_y

        zeros_x = torch.zeros(x.size()).cuda() if torch.cuda.is_available() else torch.zeros(x.size())
        zeros_y = torch.zeros(y.size()).cuda() if torch.cuda.is_available() else torch.zeros(y.size())

        loss_term_3_x = torch.max(zeros_x, output[:, 0, :, :].squeeze(dim=1))
        loss_term_3_y = torch.max(zeros_y, output[:, 1, :, :].squeeze(dim=1))
        loss_term_3 = torch.sum((loss_term_3_x + loss_term_3_y) * back_sign) / torch.sum(back_sign)

        loss = loss_term_1 - self.lamda * loss_term_2 + loss_term_3

        return loss

class MS_SSIM_Loss(MS_SSIM):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MS_SSIM_Loss, self).__init__(window_size=window_size, size_average=size_average, channel=channel)

    def forward(self, img1, img2):
        ms_ssim = super(MS_SSIM_Loss, self).forward(img1, img2)
        return 100*( 1 - ms_ssim)

class SSIM_Loss(SSIM):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM_Loss, self).__init__(window_size=window_size, size_average=size_average, val_range=val_range)

    def forward(self, img1, img2):
        ssim = super(SSIM_Loss, self).forward(img1, img2)
        return 100*( 1 - ssim)

class MS_SSIM_Loss_v2(MS_SSIM_v2):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MS_SSIM_Loss_v2, self).__init__(win_size=window_size, size_average=size_average, channel=channel, data_range=255, nonnegative_ssim=True)

    def forward(self, img1, img2):
        ms_ssim = super(MS_SSIM_Loss_v2, self).forward(img1, img2)
        return 100*( 1 - ms_ssim)

class SSIM_Loss_v2(SSIM_v2):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_Loss_v2, self).__init__(win_size=window_size, size_average=size_average, data_range=255, nonnegative_ssim=True)

    def forward(self, img1, img2):
        ssim = super(SSIM_Loss_v2, self).forward(img1, img2)
        return 100*( 1 - ssim)
