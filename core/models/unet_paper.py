"""
U-net implementation from the reported paper: https://arxiv.org/abs/1505.04597
Model follows the paper implementation, except for the use of padding in order to keep feature maps size the same.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class UnetConvBlock(nn.Module):
    """
    Constructs a UNet downsampling block

       Parameters:
            input_nc (int)      -- the number of input channels
            output_nc (int)     -- the number of output channels
            norm_layer          -- normalization layer
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            user_dropout (bool) -- if use dropout layers.
            kernel_size (int)   -- convolution kernel size
    """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, padding=0, innermost=False, dropout=0.2):
        super(UnetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(input_nc, output_nc, kernel_size=3, padding=int(padding)))
        block.append(norm_layer(output_nc))
        block.append(nn.ReLU())

        block.append(nn.Conv2d(output_nc, output_nc, kernel_size=3, padding=int(padding)))
        block.append(norm_layer(output_nc))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)
        self.innermost = innermost

    def forward(self, x):
        out = self.block(x)
        return out

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
              remove_skip (bool)  -- if skip connections should be disabled or not
      """
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, padding=1, remove_skip=False, outermost=False):
        super(UNetUpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=2, stride=2)
        self.conv_block = UnetConvBlock(output_nc * 2, output_nc, norm_layer, padding)
        self.outermost = outermost

    def forward(self, x, skip=None):
        out = self.up(x)

        if skip is not None:
            out = torch.cat([out, skip], 1)
        out = self.conv_block(out)

        return out

class UNet_paper(nn.Module):
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
            remove_skip (int [0,1])-- if skip connections should be disabled or not
            reconstruct (int [0,1])-- if we should reconstruct the next image or not
            sequence_model (str)   -- the sequence model that for the sequence mode []
            num_levels_tcn(int)    -- number of levels of the TemporalConvNet
      """

    def __init__(self, num_classes, args, norm_layer=nn.BatchNorm2d, input_nc=3):
        super(UNet_paper, self).__init__(args)

        self.num_downs = args.num_downs
        self.ngf = args.ngf

        self.encoder = self.build_encoder(self.num_downs, input_nc, self.ngf, norm_layer)
        self.decoder = self.build_decoder(self.num_downs, num_classes, self.ngf, norm_layer)
        self.decoder_last_conv = nn.Conv2d(self.ngf, num_classes, 1)

    def build_encoder(self, num_downs, input_nc, ngf, norm_layer):
        """Constructs a UNet downsampling encoder, consisting of $num_downs UNetDownBlocks

             Parameters:
                  num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                         image of size 128x128 will become of size 1x1 # at the bottleneck
                  input_nc (int)      -- the number of input channels
                  ngf (int)           -- the number of filters in the last conv layer
                  norm_layer          -- normalization layer
             Returns:
                  nn.Sequential consisting of $num_downs UnetDownBlocks
          """
        layers = []
        layers.append(UnetConvBlock(input_nc=input_nc, output_nc=ngf, norm_layer=norm_layer, padding=1))
        layers.append(UnetConvBlock(input_nc=ngf, output_nc=ngf * 2, norm_layer=norm_layer, padding=1))
        layers.append(UnetConvBlock(input_nc=ngf * 2, output_nc=ngf * 4, norm_layer=norm_layer, padding=1))
        layers.append(UnetConvBlock(input_nc=ngf * 4, output_nc=ngf * 8, norm_layer=norm_layer, padding=1))

        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            layers.append(UnetConvBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, padding=1))

        layers.append(UnetConvBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, padding=1, innermost=True))

        return nn.Sequential(*layers)

    def build_decoder(self, num_downs, num_classes, ngf, norm_layer, remove_skip=0):
        """Constructs a UNet downsampling encoder, consisting of $num_downs UNetUpBlocks

           Parameters:
                num_downs (int)     -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                       image of size 128x128 will become of size 1x1 # at the bottleneck
                num_classes (int)   -- number of classes to classify
                output_nc (int)     -- the number of output channels. outermost is ngf, innermost is ngf * 8
                norm_layer          -- normalization layer
                remove_skip (int)   -- if skip connections should be disabled or not

           Returns:
                nn.Sequential consisting of $num_downs UnetUpBlocks
        """
        layers = []
        layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, remove_skip=remove_skip))

        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 8, norm_layer=norm_layer, remove_skip=remove_skip))

        layers.append(UNetUpBlock(input_nc=ngf * 8, output_nc=ngf * 4, norm_layer=norm_layer, remove_skip=remove_skip))
        layers.append(UNetUpBlock(input_nc=ngf * 4, output_nc=ngf * 2, norm_layer=norm_layer, remove_skip=remove_skip))
        layers.append(UNetUpBlock(input_nc=ngf*2, output_nc=ngf, norm_layer=norm_layer, remove_skip=remove_skip, outermost=True))

        return nn.Sequential(*layers)

    def encoder_forward(self, x):
        skip_connections = []
        for i, down in enumerate(self.encoder):
            x = down(x)

            if not down.innermost:
                skip_connections.append(x)
                x = F.max_pool2d(x, 2)

        return x, skip_connections

    def decoder_forward(self, x, skip_connections):
        out = None
        for i, up in enumerate(self.decoder):
            skip = skip_connections.pop()
            if out is None:
                out = up(x, skip)
            else:
                out = up(out, skip)

        out = self.decoder_last_conv(out)
        return out

    def forward(self, x):
        x, skip_connections = self.encoder_forward(x)
        out = self.decoder_forward(x, skip_connections)

        return out