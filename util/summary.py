import os
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import glob

from util.general_functions import tensor2im, get_flat_images


from constants import *

class TensorboardSummary(object):

    def __init__(self, args):
        self.args = args
        self.experiment_dir = self.generate_directory(args)
        self.writer = SummaryWriter(log_dir=os.path.join(self.experiment_dir))

        self.train_step = 0
        self.test_step = 0
        self.visualization_step = 0

    def generate_directory(self, args):
        checkname = 'debug' if args.debug else ''
        checkname += args.model
        checkname += '_sc' if args.separable_conv else ''
        checkname += '-refined' if args.refine_network else ''

        if 'deeplab' in args.model:
            checkname += '-os_' + str(args.output_stride)
            checkname += '-ls_1' if args.learned_upsampling else ''
            checkname += '-pt_1' if args.pretrained else ''
            checkname += '-aspp_0' if not args.use_aspp else ''

        if 'unet' in args.model:
            checkname += '-downs_' + str(args.num_downs) + '-ngf_' + str(args.ngf) + '-type_' + str(args.down_type)

        checkname += '-loss_' + args.loss_type
        checkname += '-sloss_' if args.second_loss else ''

        if args.clip > 0:
            checkname += '-clipping_' + str(args.clip)

        if args.resize:
            checkname += '-' + ','.join([str(x) for x in list(args.resize)])
        checkname += '-epochs_' + str(args.epochs)
        checkname += '-trainval' if args.trainval else ''

        current_dir = os.path.dirname(__file__)
        directory = os.path.join(current_dir, args.results_root, args.results_dir, args.dataset_dir, args.dataset, args.model, checkname)

        runs = sorted(glob.glob(os.path.join(directory, 'experiment_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
        experiment_dir = os.path.join(directory, 'experiment_{}'.format(str(run_id)))

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        return experiment_dir

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def visualize_image(self, images, targets, outputs, split="train"):
        step = self.get_step(split)

        outputs, targets = get_flat_images(self.args.dataset, images, outputs, targets)

        images = [tensor2im(image) for image in images]
        outputs = [tensor2im(output)[:, : int(self.args.resize[0] /2), : int(self.args.resize[1] / 2)] for output in outputs]
        targets = [tensor2im(target)[:, : int(self.args.resize[0] / 2), : int(self.args.resize[1] / 2)] for target in targets]

        grid_image = make_grid(images)
        self.writer.add_image(split + '/ZZ Image', grid_image, step)

        grid_image = make_grid(outputs)
        self.writer.add_image(split + '/Predicted label', grid_image, step)

        grid_image = make_grid(targets)
        self.writer.add_image(split + '/Groundtruth label', grid_image, step)

        return images, outputs, targets

    def save_network(self, model):
        path = self.experiment_dir[self.experiment_dir.find(self.args.results_dir):].replace(self.args.results_dir, self.args.save_dir)
        if not os.path.isdir(path):
            os.makedirs(path)

        torch.save(model.state_dict(), path + '/' + 'network.pth')

    def load_network(self, model):
        path = self.args.pretrained_models_dir
        state_dict = torch.load(path) if self.args.cuda else torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        return model

    def get_step(self, split):
        if split == TRAIN:
            self.train_step += 1
            return self.train_step
        elif split == TEST:
            self.test_step += 1
            return self.test_step
        elif split == VISUALIZATION:
            self.visualization_step += 1
            return self.visualization_step