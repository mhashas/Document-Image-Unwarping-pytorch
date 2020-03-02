import cv2
from scipy import io
import random
import torch
import torchvision.transforms as standard_transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

from core.trainers.trainer import Trainer
from parser_options import ParserOptions
from util.general_functions import apply_transformation_to_image_cv, apply_transformation_to_image, invert_vector_field, get_model, make_data_loader
from constants import *

def cv2_invert(invert=True):
    deformed_label = cv2.imread('../deformed_label.jpg', cv2.IMREAD_COLOR)

    if invert:
        vector_field = io.loadmat('../fm.mat')['vector_field']
    else:
        vector_field = io.loadmat('../fm.mat')['inverted_vector_field']

    flatten_label = apply_transformation_to_image_cv(deformed_label, vector_field, invert=invert)
    cv2.imwrite('../our_flatten.jpg', flatten_label)

def pytorch_invert(invert=False):
    deformed_label = cv2.imread('../deformed_label.jpg', cv2.IMREAD_COLOR)
    deformed_label = standard_transforms.ToTensor()(deformed_label).unsqueeze(dim=0)

    if invert:
        vector_field = io.loadmat('../fm.mat')['vector_field']
        vector_field = invert_vector_field(vector_field)
    else:
        vector_field = io.loadmat('../fm.mat')['inverted_vector_field']

    vector_field = torch.Tensor(vector_field).unsqueeze(dim=0)
    vector_field = vector_field.permute(0, 3, 1, 2)
    flatten_label = apply_transformation_to_image(deformed_label, vector_field)
    plt.imsave('../our_flatten_2.jpg', flatten_label.permute(1,2,0).cpu().numpy())

def check_duplicates(source_folder_name, destination_folder_name):
    source_files = set(os.listdir(source_folder_name))
    destination_files = set(os.listdir(destination_folder_name))
    intersection = source_files.intersection(destination_files)
    intersection.remove('Thumbs.db')

    if len(intersection) == 0:
        print("OK")
    else:
        print("NOT OK")

def network_predict(iterations=51, pretrained_model=''):
    if not pretrained_model:
        raise NotImplementedError()

    args = ParserOptions().parse()
    args.cuda = False
    args.batch_size = 1
    args.inference = 1
    args.pretrained_models_dir = pretrained_model
    args.num_downs = 8
    args.resize, args.size = (256,256), (256,256)
    args.model = DEEPLAB_50
    #args.refine_network = 1
    trainer = Trainer(args)
    mean_time = trainer.calculate_inference_speed(iterations)
    print('Mean time', mean_time)

if __name__ == "__main__":
    model = 'saved_models/deeplab_50.pth'
    network_predict(pretrained_model=model)