import os
from PIL import Image
from torch.utils import data
import torchvision.transforms as standard_transforms
import util.custom_transforms as custom_transforms
from scipy import io


class InvertedDocunet(data.Dataset):
    NUM_CLASSES = 2
    CLASSES = ["foreground", "background"]
    ROOT = '../../'
    DEFORMED = 'deformed_labels'
    DEFORMED_EXT = '.jpg'
    VECTOR_FIELD = 'inverted_vf'
    VECTOR_FIELD_EXT = '.mat'

    def __init__(self, args, split="train"):
        self.args = args
        self.split = split
        self.dataset = self.make_dataset()

        if len(self.dataset) == 0:
            raise RuntimeError('Found 0 images, please check the dataset')

        self.transform = self.get_transforms()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_path, label_path = self.dataset[index]
        image = Image.open(image_path)
        label = io.loadmat(label_path)['inverted_vector_field']

        if self.transform is not None:
            image = self.transform(image)

        image, label = standard_transforms.ToTensor()(image), standard_transforms.ToTensor()(label)

        label = label.float()
        return image, label

    def make_dataset(self):
        current_dir = os.path.dirname(__file__)
        path_image = os.path.join(current_dir, self.ROOT, self.args.dataset_dir, self.split, self.DEFORMED + '_' + 'x'.join(map(str, self.args.size)))
        path_label = os.path.join(current_dir, self.ROOT, self.args.dataset_dir, self.split, self.VECTOR_FIELD + '_' + 'x'.join(map(str, self.args.size)))

        images_path = os.listdir(path_image)
        images_path = [image_path for image_path in images_path if image_path.endswith(self.DEFORMED_EXT)]
        items = []

        for i in range(len(images_path)):
            image_path = images_path[i]
            label_path = image_path.replace(self.DEFORMED_EXT, self.VECTOR_FIELD_EXT)
            items.append((os.path.join(path_image, image_path), os.path.join(path_label, label_path)))

        return items

    def get_transforms(self):
        return None




