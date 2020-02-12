'''
Adapted work from https://github.com/jfzhang95/pytorch-deeplab-xception
'''

import torch
import torch.nn.functional as F
import torch.nn as nn

from core.models.resnet import ResNet101, ResNet50, ResNet34, ResNet18
from core.models.mobilenetv2 import MobileNet_v2
from core.models.mobilenet_v2_dilation import MobileNet_v2_dilation
from constants import *

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer=nn.BatchNorm2d):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = norm_layer(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, output_stride, norm_layer=nn.BatchNorm2d, inplanes=2048):
        super(ASPP, self).__init__()

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], norm_layer=norm_layer)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], norm_layer=norm_layer)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], norm_layer=norm_layer)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], norm_layer=norm_layer)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             norm_layer(256),
                                             nn.ReLU())

        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = norm_layer(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)

        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

class Decoder(nn.Module):
    def __init__(self, num_classes, norm_layer=nn.BatchNorm2d, inplanes=256, aspp_outplanes=256):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, 48, 1, bias=False)
        self.bn1 = norm_layer(48)
        self.relu = nn.ReLU()

        inplanes = 48 + aspp_outplanes

        self.last_conv = nn.Sequential(nn.Conv2d(inplanes, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(256),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_layer(256),
                                       nn.ReLU(),
                                       nn.Dropout2d(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

class DeepLabv3_plus(nn.Module):
    def __init__(self, args, num_classes=21, norm_layer=nn.BatchNorm2d, input_channels=3):
        super(DeepLabv3_plus, self).__init__()
        self.args = args

        if args.model == DEEPLAB:
            self.backbone = ResNet101(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained, input_channels=input_channels)
            self.aspp_inplanes = 2048
            self.decoder_inplanes = 256

            if self.args.refine_network:
                self.refine_backbone = ResNet101(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained, input_channels=input_channels + num_classes)
        elif args.model == DEEPLAB_50:
            self.backbone = ResNet50(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained)
            self.aspp_inplanes = 2048
            self.decoder_inplanes = 256

            if self.args.refine_network:
                self.refine_backbone = ResNet50(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained, input_channels=input_channels + num_classes)
        elif args.model == DEEPLAB_34:
            self.backbone = ResNet34(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained)
            self.aspp_inplanes = 2048
            self.decoder_inplanes = 256

            if self.args.refine_network:
                self.refine_backbone = ResNet34(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained, input_channels=input_channels + num_classes)
        elif args.model == DEEPLAB_18:
            self.backbone = ResNet18(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained)
            self.aspp_inplanes = 2048
            self.decoder_inplanes = 256

            if self.args.refine_network:
                self.refine_backbone = ResNet18(args.output_stride, norm_layer=norm_layer, pretrained=args.pretrained, input_channels=input_channels + num_classes)
        elif args.model == DEEPLAB_MOBILENET:
            self.backbone = MobileNet_v2(pretrained=args.pretrained, first_layer_input_channels=input_channels)
            self.aspp_inplanes = 320
            self.decoder_inplanes = 24

            if self.args.refine_network:
                self.refine_backbone = MobileNet_v2(pretrained=args.pretrained, first_layer_input_channels=input_channels + num_classes)

        elif args.model == DEEPLAB_MOBILENET_DILATION:
            self.backbone = MobileNet_v2_dilation(pretrained=args.pretrained, first_layer_input_channels=input_channels)
            self.aspp_inplanes = 320

            if self.args.refine_network:
                self.refine_backbone = MobileNet_v2_dilation(pretrained=args.pretrained, first_layer_input_channels=input_channels + num_classes)
            self.decoder_inplanes = 24
        else:
            raise NotImplementedError

        if self.args.use_aspp:
            self.aspp = ASPP(args.output_stride, norm_layer=norm_layer, inplanes=self.aspp_inplanes)

        aspp_outplanes = 256 if self.args.use_aspp else self.aspp_inplanes
        self.decoder = Decoder(num_classes, norm_layer=norm_layer, inplanes=self.decoder_inplanes, aspp_outplanes=aspp_outplanes)

        if self.args.learned_upsampling:
            self.learned_upsampling = nn.Sequential(nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
                                                    nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1))

        if self.args.refine_network:
            if self.args.use_aspp:
                self.refine_aspp = ASPP(args.output_stride, norm_layer=norm_layer, inplanes=self.aspp_inplanes)

            self.refine_decoder = Decoder(num_classes, norm_layer=norm_layer, inplanes=self.decoder_inplanes, aspp_outplanes=aspp_outplanes)

            if self.args.learned_upsampling:
                self.refine_learned_upsampling = nn.Sequential(nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1),
                                                               nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1))


    def forward(self, input):
        output, low_level_feat = self.backbone(input)

        if self.args.use_aspp:
            output = self.aspp(output)

        output = self.decoder(output, low_level_feat)

        if self.args.learned_upsampling:
            output = self.learned_upsampling(output)
        else:
            output = F.interpolate(output, size=input.size()[2:], mode='bilinear', align_corners=True)

        if self.args.refine_network:
            second_output, low_level_feat = self.refine_backbone(torch.cat((input, output), dim=1))

            if self.args.use_aspp:
                second_output = self.refine_aspp(second_output)

            second_output = self.refine_decoder(second_output, low_level_feat)

            if self.args.learned_upsampling:
                second_output = self.refine_learned_upsampling(second_output)
            else:
                second_output = F.interpolate(second_output, size=input.size()[2:], mode='bilinear', align_corners=True)

            return output, second_output

        return output
    def get_train_parameters(self, lr):
        train_params = [{'params': self.parameters(), 'lr': lr}]

        return train_params