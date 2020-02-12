import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import numpy as np
import functools
import cv2

from core.models.deeplabv3_plus import DeepLabv3_plus
from core.models.mobilenetv2 import MobileNet_v2
from core.models.pspnet import PSPNet
from core.models.unet import UNet
from core.models.unet_paper import UNet_paper
from core.models.unet_pytorch import UNet_torch

from dataloader.docunet import Docunet
from dataloader.docunet_inverted import InvertedDocunet
from dataloader.docunet_im2im import DocunetIm2Im

from util.losses import *
from constants import *

def make_data_loader(args, split=TRAIN):
    """
    Builds the model based on the provided arguments

        Parameters:
        args (argparse)    -- input arguments
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
    """
    if args.dataset == DOCUNET:
        dataset = Docunet
    elif args.dataset == DOCUNET_INVERTED:
        dataset = InvertedDocunet
    elif args.dataset == DOCUNET_IM2IM:
        dataset = DocunetIm2Im
    else:
        raise NotImplementedError

    if split == TRAINVAL:
        train_set = dataset(args, split=TRAIN)
        val_set = dataset(args, split=VAL)
        trainval_set = ConcatDataset([train_set, val_set])
        loader = DataLoader(trainval_set, batch_size=args.batch_size, num_workers=1, shuffle=True)
    else:
        set = dataset(args, split=split)

        if split == TRAIN:
            loader = DataLoader(set, batch_size=args.batch_size, num_workers=1, shuffle=True)
        else:
            loader = DataLoader(set, batch_size=args.batch_size, num_workers=1, shuffle=False)

    return loader

def get_model(args):
    """
    Builds the model based on the provided arguments and returns the initialized model

        Parameters:
        args (argparse)    -- command line arguments
    """

    norm_layer = get_norm_layer(args.norm_layer)
    num_classes = get_num_classes(args.dataset)

    if DEEPLAB in args.model:
        model = DeepLabv3_plus(args, num_classes=num_classes, norm_layer=norm_layer)

        if args.model != DEEPLAB_MOBILENET and args.separable_conv:
            convert_to_separable_conv(model)

        model = init_model(model, args.init_type)
    elif UNET in args.model:
        if args.model == UNET:
            model = UNet(num_classes=num_classes, args=args, norm_layer=norm_layer)
        elif args.model == UNET_PAPER:
            model = UNet_paper(num_classes=num_classes, args=args, norm_layer=norm_layer)
        elif args.model == UNET_PYTORCH:
            model = UNet_torch(num_classes=num_classes, args=args)

        if args.separable_conv:
            convert_to_separable_conv(model)

        model = init_model(model, args.init_type)
    elif PSPNET in args.model:
        model = PSPNet(num_classes=num_classes, args=args)
    else:
        raise NotImplementedError

    print("Built " + args.model)

    if args.cuda:
        model = model.cuda()

    return model

def convert_to_separable_conv(module):
    class SeparableConvolution(nn.Module):
        """ Separable Convolution
        """

        def __init__(self, in_channels, out_channels, kernel_size,
                     stride=1, padding=0, dilation=1, bias=True):
            super(SeparableConvolution, self).__init__()
            self.body = nn.Sequential(
                # Separable Conv
                nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                          dilation=dilation, bias=bias, groups=in_channels),
                # PointWise Conv
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
            )

        def forward(self, x):
            return self.body(x)

    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = SeparableConvolution(module.in_channels,
                                      module.out_channels,
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias is not None)

    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module

def get_loss_function(mode):
    if mode == DOCUNET_LOSS:
        loss = DocunetLoss()
    elif mode == MS_SSIM_LOSS:
        loss = MS_SSIM_Loss()
    elif mode == MS_SSIM_LOSS_V2:
        loss = MS_SSIM_Loss_v2()
    elif mode == SSIM_LOSS:
        loss = SSIM_Loss()
    elif mode == L1_LOSS:
        loss = torch.nn.L1Loss()
    elif mode == SMOOTH_L1_LOSS:
        loss = torch.nn.SmoothL1Loss()
    elif mode == MSE_LOSS:
        loss = torch.nn.MSELoss()
    else:
        raise NotImplementedError

    return loss

def get_num_classes(dataset):
    if dataset == DOCUNET or dataset == DOCUNET_INVERTED:
        num_classes = 2
    elif dataset == DOCUNET_IM2IM:
        num_classes = 3
    else:
        raise NotImplementedError

    return num_classes


def get_optimizer(model, args):
    """
    Builds the optimizer for the model based on the provided arguments and returns the optimizer

        Parameters:
        model          -- the network to be optimized
        args           -- command line arguments
    """
    if args.gpu_ids:
        train_params = model.module.get_train_parameters(args.lr)
    else:
        train_params = model.get_train_parameters(args.lr)

    if args.optim == SGD:
        optimizer = optim.SGD(train_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
    elif args.optim == ADAM:
        optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == AMSGRAD:
        optimizer = optim.Adam(train_params, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    else:
        raise NotImplementedError

    return optimizer


def get_norm_layer(norm_type=INSTANCE_NORM):
    """Returns a normalization layer

        Parameters:
            norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """

    if norm_type == BATCH_NORM:
        norm_layer = nn.BatchNorm2d
    elif norm_type == INSTANCE_NORM:
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_model(net, init_type=NORMAL_INIT, init_gain=0.02):
    """Initialize the network weights

    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.

    Return an initialized network.
    """

    init_weights(net, init_type, init_gain=init_gain)
    return net

def init_weights(net, init_type=NORMAL_INIT, init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == NORMAL_INIT:
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == XAVIER_INIT:
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == KAIMING_INIT:
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='leaky_relu')
            elif init_type == ORTHOGONAL_INIT:
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, '_all_weights') and (classname.find('LSTM') != -1 or classname.find('GRU') != -1):
            for names in m._all_weights:
                for name in filter(lambda n: "weight" in n, names):
                    weight = getattr(m, name)
                    nn.init.xavier_normal_(weight.data, gain=init_gain)

                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(m, name)
                    nn.init.constant_(bias.data, 0.0)

                    if classname.find('LSTM') != -1:
                        n = bias.size(0)
                        start, end = n // 4, n // 2
                        nn.init.constant_(bias.data[start:end], 1.)
        elif classname.find('BatchNorm2d') != -1 or classname.find('SynchronizedBatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('Initialized network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def set_requires_grad(net, requires_grad=False):
    """Set requies_grad=False for the network to avoid unnecessary computations
    Parameters:
        net (network)
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if net is not None:
        for param in net.parameters():
            param.requires_grad = requires_grad

def tensor2im(input_image, imtype=np.uint8, return_tensor=True):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.ndim == 3:
            image_numpy = (image_numpy - np.min(image_numpy))/(np.max(image_numpy)-np.min(image_numpy))
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image

    return torch.from_numpy(image_numpy.astype(imtype)) if return_tensor else np.transpose(image_numpy, (1,2,0))

def get_flat_images(dataset, images, outputs, targets):
    if dataset == DOCUNET:
        pass
    elif dataset == DOCUNET_INVERTED:
        outputs = apply_transformation_to_image(images, outputs)
        targets = apply_transformation_to_image(images, targets)
    else:
        pass

    return outputs, targets

def apply_transformation_to_image(img, vector_field):
    vector_field = scale_vector_field_tensor(vector_field)
    vector_field = vector_field.permute(0, 2, 3, 1)
    flatten_image = nn.functional.grid_sample(img, vector_field, mode='bilinear', align_corners=True)

    return flatten_image.squeeze()

def apply_transformation_to_image_cv(img, vector_field, invert=False):
    if invert:
        vector_field = invert_vector_field(vector_field)

    map_x = vector_field[:, :, 0]
    map_y = vector_field[:, :, 1]
    transformed_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return transformed_img

def invert_vector_field(vector_field):
    vector_field_x = vector_field[:, :, 0]
    vector_field_y = vector_field[:, :, 1]

    assert(vector_field_x.shape == vector_field_y.shape)
    rows = vector_field_x.shape[0]
    cols = vector_field_x.shape[1]

    m_x = np.ones(vector_field_x.shape, dtype=vector_field_x.dtype) * -1
    m_y = np.ones(vector_field_y.shape, dtype=vector_field_y.dtype) * -1
    for i in range(rows):
        for j in range(cols):
            i_ = int(round(vector_field_y[i, j]))
            j_ = int(round(vector_field_x[i, j]))
            if 0 <= i_ < rows and 0 <= j_ < cols:
                m_x[i_, j_] = j
                m_y[i_, j_] = i
    return np.stack([m_x, m_y], axis=2)


def scale_vector_field_tensor(vector_field):
    vector_field = torch.where(vector_field < 0, torch.tensor(3 * vector_field.shape[3], dtype=vector_field.dtype, device=vector_field.device), vector_field)
    vector_field = (vector_field / (vector_field.shape[3] / 2)) - 1

    return vector_field

def print_training_info(args):
    print('Dataset', args.dataset)

    if 'unet' in args.model:
        print('Ngf', args.ngf)
        print('Num downs', args.num_downs)
        print('Down type', args.down_type)

    if 'deeplab' in args.model:
        print('Output stride', args.output_stride)
        print('Learned upsampling', args.learned_upsampling)
        print('Pretrained', args.pretrained)
        print('Use aspp', args.use_aspp)

    print('Refine network', args.refine_network)
    print('Separable conv', args.separable_conv)
    print('Optimizer', args.optim)
    print('Learning rate', args.lr)
    print('Second loss', args.second_loss)

    if args.clip > 0:
        print('Gradient clip', args.clip)

    print('Resize', args.resize)
    print('Batch size', args.batch_size)
    print('Norm layer', args.norm_layer)
    print('Using cuda', args.cuda)
    print('Using ' + args.loss_type + ' loss')
    print('Starting Epoch:', args.start_epoch)
    print('Total Epoches:', args.epochs)



