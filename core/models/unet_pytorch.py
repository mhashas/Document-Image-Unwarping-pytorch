from collections import OrderedDict

import torch
import torch.nn as nn

class UNet_torch(nn.Module):

    def __init__(self, num_classes=1, args=None, in_channels=3):
        super(UNet_torch, self).__init__()
        self.ngf = args.ngf

        self.encoder1 = UNet_torch._block(in_channels, self.ngf, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNet_torch._block(self.ngf, self.ngf * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNet_torch._block(self.ngf * 2, self.ngf * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = UNet_torch._block(self.ngf * 4, self.ngf * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet_torch._block(self.ngf * 8, self.ngf * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            self.ngf * 16, self.ngf * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet_torch._block((self.ngf * 8) * 2, self.ngf * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            self.ngf * 8, self.ngf * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet_torch._block((self.ngf * 4) * 2, self.ngf * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            self.ngf * 4, self.ngf * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet_torch._block((self.ngf * 2) * 2, self.ngf * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            self.ngf * 2, self.ngf, kernel_size=2, stride=2
        )
        self.decoder1 = UNet_torch._block(self.ngf * 2, self.ngf, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=self.ngf, out_channels=num_classes, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        final = self.conv(dec1)
        return final

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                ]
            )
        )

    def get_train_parameters(self, lr):
        params = [{'params': self.parameters(), 'lr': lr}]

        return params