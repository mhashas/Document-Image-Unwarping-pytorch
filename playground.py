import cv2
from scipy import io
import random
import torch
import torchvision.transforms as standard_transforms
import matplotlib.pyplot as plt
import time

from util.general_functions import apply_transformation_to_image_cv, apply_transformation_to_image, invert_vector_field

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


if __name__ == "__main__":
    pytorch_invert()