import os
from PIL import Image
from torch.utils import data
import torchvision.transforms as standard_transforms
import util.custom_transforms as custom_transforms
from scipy import io


class DocunetIm2Im(data.Dataset):
    NUM_CLASSES = 2
    CLASSES = ["foreground", "background"]
    ROOT = '../../../datasets/'
    DEFORMED = 'deformed_labels'
    LABELS = 'cropped_labels'
    DEFORMED_EXT = '.jpg'
    LABEL_EXT = '.jpg'

    def __init__(self, args, split="train"):
        self.args = args
        self.split = split
        self.dataset = self.make_dataset()

        if len(self.dataset) == 0:
            raise RuntimeError('Found 0 images, please check the dataset')

        self.joint_transform = self.get_transforms()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path, label_path = self.dataset[index]
        image = Image.open(image_path)
        label = Image.open(label_path)

        if self.joint_transform is not None:
            image, label = self.joint_transform(image, label)

        image, label = standard_transforms.ToTensor()(image), standard_transforms.ToTensor()(label)

        return image, label

    def make_dataset(self):
        current_dir = os.path.dirname(__file__)
        images_path = os.path.join(current_dir, self.ROOT, self.args.dataset_dir, self.split, self.DEFORMED + '_' + 'x'.join(map(str, self.args.size)))
        labels_path = os.path.join(current_dir, self.ROOT, self.args.dataset_dir, self.split, self.LABELS)

        images_name = os.listdir(images_path)
        images_name = [image_name for image_name in images_name if image_name.endswith(self.DEFORMED_EXT)]
        items = []

        for i in range(len(images_name)):
            image_name = images_name[i]
            label_name = '_'.join(image_name.split('_')[:-1]) + self.LABEL_EXT
            items.append((os.path.join(images_path, image_name), os.path.join(labels_path, label_name)))

        return items

    def get_transforms(self):
        if self.split == 'train':
            joint = custom_transforms.Compose([
                custom_transforms.Resize(self.args.resize),
                custom_transforms.RandomHorizontallyFlip(),
                # custom_transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                custom_transforms.RandomGaussianBlur()
            ])
        elif self.split == 'val' or self.split == 'test' or self.split == 'demoVideo':
            joint = custom_transforms.Compose([
                custom_transforms.Resize(self.args.resize),
            ])
        else:
            raise RuntimeError('Invalid dataset mode')

        return joint




