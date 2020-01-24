import argparse
import torch
import os
import math

from constants import *

class ParserOptions():
    """This class defines options that are used by the program"""

    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Semantic Video Segmentation training')

        # model specific
        parser.add_argument('--model', type=str, default=UNET, choices=[DEEPLAB, DEEPLAB_50, UNET, UNET_PAPER, UNET_PYTORCH, PSPNET], help='model name (default:' + DEEPLAB + ')')
        parser.add_argument('--refine_network', type=int, default=0, choices=[0,1], help='if we should refine the first prediction with a second network ')
        parser.add_argument('--dataset', type=str, default=DOCUNET_INVERTED, choices=[DOCUNET, DOCUNET_IM2IM, DOCUNET_INVERTED], help='dataset (default:' + DOCUNET + ')')
        parser.add_argument('--dataset_dir', type=str, default='labels_dataset', help='name of the dir in which the dataset is located')
        parser.add_argument('--loss_type', type=str, default=SSIM_LOSS, choices=[DOCUNET_LOSS, SSIM_LOSS, SSIM_LOSS_V2, MS_SSIM_LOSS, MS_SSIM_LOSS_V2, L1_LOSS, SMOOTH_L1_LOSS, MSE_LOSS], help='loss func type (default:' + DOCUNET_LOSS + ')')
        parser.add_argument('--second_loss', type=int, default=1, choices=[0,1], help='if we should use two losses')
        parser.add_argument('--second_loss_rate', type=float, default=10, help='used to tune the overall impact of the second loss')
        parser.add_argument('--norm_layer', type=str, default=BATCH_NORM, choices=[INSTANCE_NORM, BATCH_NORM, SYNC_BATCH_NORM])
        parser.add_argument('--init_type', type=str, default=NORMAL_INIT, choices=[NORMAL_INIT, KAIMING_INIT, XAVIER_INIT, ORTHOGONAL_INIT])
        parser.add_argument('--resize', type=str, default='256,256', help='image resize: h,w')
        parser.add_argument('--batch_size', type=int, default=2, metavar='N', help='input batch size for training (default: 2)')
        parser.add_argument('--optim', type=str, default=ADAM, choices=[SGD, ADAM, RMSPROP, AMSGRAD, ADABOUND])
        parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: auto)')
        parser.add_argument('--lr_policy', type=str, default='poly', choices=['poly', 'step', 'cos', 'linear'], help='lr scheduler mode: (default: poly)')
        parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
        parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--clip', type=float, default=0, help='gradient clip, 0 means no clip (default: 0)')

        # training specific
        parser.add_argument('--size', type=str, default='1024,1024', help='image size: h,w')
        parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='starting epoch')
        parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: auto)')
        parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval (default: 1)')
        parser.add_argument('--trainval', type=int, default=0, choices=[0,1], help='determines whether whe should use validation images as well for training')
        parser.add_argument('--inference', type=int, default=0, choices=[0,1], help='if we should run the model in inference mode')
        parser.add_argument('--debug', type=int, default=1)
        parser.add_argument('--results_root', type=str, default='..')
        parser.add_argument('--results_dir', type=str, default='results_final', help='models are saved here')
        parser.add_argument('--save_dir', type=str, default='saved_models')
        parser.add_argument('--save_best_model', type=int, default=0, choices=[0,1], help='keep track of best model')
        parser.add_argument('--pretrained_models_dir', type=str, default='pretrained_models', help='root dir of the pretrained models location')

        # deeplab specific
        parser.add_argument('--output_stride', type=int, default=16, help='network output stride (default: 16)')

        # unet specific
        parser.add_argument('--num_downs', type=int, default=8, help='number of unet encoder-decoder blocks')
        parser.add_argument('--ngf', type=int, default=128, help='# of gen filters in the last conv layer')
        parser.add_argument('--down_type', type=str, default=MAXPOOL, choices=[STRIDECONV, MAXPOOL], help='method to reduce feature map size')
        parser.add_argument('--dropout', type=float, default=0.2)

        # gan specific
        parser.add_argument('--gan_training', type=int, default=0, choices=[0,1], help="if we are training in adversarial fashion or not")
        parser.add_argument('--model_discriminator', type=str, default=PIXEL_DISCRIMINATOR, choices=[PIXEL_DISCRIMINATOR], help='type of discriminator used')
        parser.add_argument('--gan_loss_type', type=str, default=LS_GAN_LOSS, choices=[LS_GAN_LOSS], help='loss function for gan')

        args = parser.parse_args()
        args.size = tuple([int(x) for x in args.size.split(',')])
        args.resize = tuple([int(x) for x in args.resize.split(',')])

        if args.debug:
            args.results_dir = 'results_dummy'

        args.num_downs = int(math.log(args.resize[0])/math.log(2))
        args.cuda = torch.cuda.is_available()
        args.gpu_ids = os.environ['CUDA_VISIBLE_DEVICES'] if ('CUDA_VISIBLE_DEVICES' in os.environ) else ''
        args.gpu_ids = list(range(len(args.gpu_ids.split(',')))) if (',' in args.gpu_ids and args.cuda) else None

        if args.gpu_ids and args.norm_layer == BATCH_NORM:
            args.norm_layer = SYNC_BATCH_NORM

        if args.gan_training:
            args.dataset = DOCUNET_IM2IM
            args.down_type = STRIDECONV

        if args.dataset == DOCUNET or args.dataset == DOCUNET_INVERTED:
            args.size = args.resize

        if args.dataset == DOCUNET_IM2IM and args.loss_type == DOCUNET_LOSS:
            args.loss_type = MS_SSIM_LOSS
            args.double_loss = 0

        self.args = args

    def parse(self):
        return self.args