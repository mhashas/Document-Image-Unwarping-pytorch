"""
Lighter U-net implementation that achieves same performance as the one reported in the paper: https://arxiv.org/abs/1505.04597
Main differences:
    a) U-net downblock has only 1 convolution instead of 2
    b) U-net upblock has only 1 convolution instead of 3
"""

import torch
import torch.nn as nn
from constants import *

class UNetDownBlock(nn.Module):
    """
    Constructs a UNet downsampling block

       Parameters:
            input_nc (int)      -- the number of input channels
            output_nc (int)     -- the number of output channels
            norm_layer (str)    -- normalization layer
            down_type (str)     -- if we should use strided convolution or maxpool for reducing the feature map
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            user_dropout (bool) -- if use dropout layers.
            kernel_size (int)   -- convolution kernel size
            bias (boolean)      -- if convolution should use bias
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, down_type=STRIDECONV, outermost=False, innermost=False, dropout=0.2, kernel_size=4, bias=True):
        super(UNetDownBlock, self).__init__()
        self.innermost = innermost
        self.outermost = outermost
        self.use_maxpool = down_type == MAXPOOL

        stride = 1 if self.use_maxpool else 2
        kernel_size = 3 if self.use_maxpool else 4
        self.conv = nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, stride=stride, padding=1, bias=bias)
        self.relu = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(2)
        self.norm = norm_layer(output_nc)
        self.dropout = nn.Dropout2d(dropout) if dropout else None

    def forward(self, x):
        if self.outermost:
            x = self.conv(x)
            x = self.norm(x)
        elif self.innermost:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
        else:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
            x = self.norm(x)

        return x

class UNetUpBlock(nn.Module):
    """
      Constructs a UNet upsampling block

         Parameters:
              input_nc (int)      -- the number of input channels
              output_nc (int)     -- the number of output channels
              norm_layer          -- normalization layer
              outermost (bool)    -- if this module is the outermost module
              innermost (bool)    -- if this module is the innermost module
              user_dropout (bool) -- if use dropout layers.
              kernel_size (int)   -- convolution kernel size
      """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, outermost=False, innermost=False, dropout=0.2, kernel_size=4, use_bias=True):
        super(UNetUpBlock, self).__init__()
        self.innermost = innermost
        self.outermost = outermost
        upconv_inner_nc = input_nc * 2

        if self.innermost:
            self.conv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1, bias=use_bias)
        elif self.outermost:
            self.conv = nn.ConvTranspose2d(upconv_inner_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.conv = nn.ConvTranspose2d(upconv_inner_nc, output_nc, kernel_size=kernel_size, stride=2, padding=1, bias=use_bias)

        self.norm = norm_layer(output_nc)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout) if dropout else None

    def forward(self, x):
        if self.outermost:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
        elif self.innermost:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
            x = self.norm(x)
        else:
            x = self.relu(x)
            if self.dropout: x = self.dropout(x)
            x = self.conv(x)
            x = self.norm(x)

        return x

class UNet(nn.Module):
    """Create a Unet-based Fully Convolutional Network
          X -------------------identity----------------------
          |-- downsampling -- |submodule| -- upsampling --|

        Parameters:
            num_classes (int)      -- the number of channels in output images
            norm_layer             -- normalization layer
            input_nc               -- number of channels of input image

            Args:
            mode (str)             -- process single frames or sequence of frames
            timesteps (int)        --
            num_downs (int)        -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                      image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)              -- the number of filters in the last conv layer
            reconstruct (int [0,1])-- if we should reconstruct the next image or not
            sequence_model (str)   -- the sequence model that for the sequence mode []
            num_levels_tcn(int)    -- number of levels of the TemporalConvNet
      """

    def __init__(self, num_classes, args, norm_layer=nn.BatchNorm2d, input_nc=3):
        super(UNet, self).__init__()

        self.refine_network = args.refine_network
        self.num_downs = args.num_downs
        self.ngf = args.ngf

        self.encoder = self.build_encoder(self.num_downs, input_nc, self.ngf, norm_layer, down_type=args.down_type)
        self.decoder = self.build_decoder(self.num_downs, num_classes, self.ngf, norm_layer)

        if self.refine_network:
            self.refine_encoder = self.build_encoder(self.num_downs, input_nc + num_classes, self.ngf, norm_layer, down_type=args.down_type)
            self.refine_decoder = self.build_decoder(self.num_downs, num_classes, self.ngf, norm_layer)

    def build_encoder(self, num_downs, input_nc, ngf, norm_layer, down_type=STRIDECONV):
        """Constructs a UNet downsampling encoder, consisting of $num_downs UNetDownBlocks
            
             Parameters:
                  num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                         image of size 128x128 will become of size 1x1 # at the bottleneck
                  input_nc (int)      -- the number of input channels
                  ngf (int)           -- the number of filters in the last conv layer
                  norm_layer (str)    -- normalization layer
                  down_type (str)     -- if we should use strided convolution or maxpool for reducing the feature map
             Returns:
                  nn.Sequential consisting of $num_downs UnetDownBlocks
        """
        layers = []
        layers.append(UNetDownBlock(input_nc=input_nc, output_nc=ngf, norm_layer=norm_layer, down_type=down_type, outermost=True))
        layers.append(UNetDownBlock(input_nc=ngf, output_nc=ngf*2, norm_layer=norm_layer, down_type=down_type))
        layers.append(UNetDownBlock(input_nc=ngf*2, output_nc=ngf*4, norm_layer=norm_layer, down_type=down_type))
        layers.append(UNetDownBlock(input_nc=ngf*4, output_nc=ngf*8, norm_layer=norm_layer, down_type=down_type))

        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            layers.append(UNetDownBlock(input_nc=ngf*8, output_nc=ngf*8, norm_layer=norm_layer, down_type=down_type))

        layers.append(UNetDownBlock(input_nc=ngf*8, output_nc=ngf*8, norm_layer=norm_layer, down_type=down_type, innermost=True))

        return nn.Sequential(*layers)

    def build_decoder(self, num_downs, num_classes, ngf, norm_layer):
        """Constructs a UNet downsampling encoder, consisting of $num_downs UNetUpBlocks

           Parameters:
                num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                       image of size 128x128 will become of size 1x1 # at the bottleneck
                num_classes (int)   -- number of classes to classify
                output_nc (int)     -- the number of output channels. outermost is ngf, innermost is ngf * 8
                norm_layer          -- normalization layer

           Returns:
                nn.Sequential consisting of $num_downs UnetUpBlocks
        """
        layers = []
        layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, innermost=True))

        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer))

        layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 4, norm_layer=norm_layer))
        layers.append(UNetUpBlock(input_nc=ngf * 4, output_nc=ngf * 2, norm_layer=norm_layer))
        layers.append(UNetUpBlock(input_nc=ngf*2, output_nc=ngf, norm_layer=norm_layer))
        layers.append(UNetUpBlock(input_nc=ngf, output_nc=num_classes, norm_layer=norm_layer, outermost=True))

        return nn.Sequential(*layers)

    def encoder_forward(self, x, use_refine_network=False):
        skip_connections = []
        model = self.refine_encoder if use_refine_network else self.encoder

        for i, down in enumerate(model):
            x = down(x)
            if down.use_maxpool:
                x = down.maxpool(x)

            if not down.innermost:
                skip_connections.append(x)

        return x, skip_connections

    def decoder_forward(self, x, skip_connections, use_refine_network=False):
        model = self.refine_decoder if use_refine_network else self.decoder

        for i, up in enumerate(model):
            if not up.innermost:
                skip = skip_connections[-i]
                out = torch.cat([skip, out], 1)
                out = up(out)
            else:
                out = up(x)

        return out

    def forward(self, x):
        output, skip_connections = self.encoder_forward(x)
        output = self.decoder_forward(output, skip_connections)

        if self.refine_network:
            second_output, skip_connections = self.encoder_forward(torch.cat((x, output), dim=1), use_refine_network=True)
            second_output = self.decoder_forward(second_output, skip_connections, use_refine_network=True)

            return output, second_output

        return output

    def get_train_parameters(self, lr):
        params = [{'params': self.parameters(), 'lr': lr}]

        return params