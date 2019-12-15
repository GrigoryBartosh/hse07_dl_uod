import os

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class EmojiDataset(Dataset):

    def __init__(self, root, transform):
        self.root_dir = root
        self.transform = transform
        self.len = len(os.listdir(self.root_dir))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        img_name = os.path.join(self.root_dir, str(idx) + '.png')
        sample = Image.open(img_name)

        sample = self.transform(sample)[:3]

        return sample


class MovableDataset(Dataset):

    def __init__(self, root, transform):
        self.root_dir = root
        self.emoji_dir = './../../datasets/emoji/'
        self.image_dir = os.path.join(self.root_dir, 'data')
        self.info_dir = os.path.join(self.root_dir, 'info')
        self.transform = transform
        self.len = len(os.listdir(self.image_dir))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, str(idx) + '.png')
        info_name = os.path.join(self.info_dir, str(idx) + '.txt')
        y = Image.open(img_name)
        y = self.transform(y)[:3]
        y = y.squeeze()

        info = []
        with open(info_name) as in_file:
            for index, line in enumerate(in_file.readlines()[1:]):
                if index > 1:
                    raise ValueError('Must be only one emoji on the image')
                info = torch.tensor(list(map(float, line.split(' '))))
        x_moves = info[:5]
        x_probs = info[5:]
        x_class_number = x_probs.argmax()
        x_image_dir = os.path.join(self.emoji_dir, f'test_{x_class_number.item()}.png')
        x_image = Image.open(x_image_dir)
        x_image = x_image.resize((320, 320))
        x_image = np.array(x_image)
        x_image[x_image[:, :, 3] <= 15] = 255
        x_image = Image.fromarray(x_image)
        x_image = self.transform(x_image)
        x_image = x_image.squeeze()

        return (x_image, x_moves), y
