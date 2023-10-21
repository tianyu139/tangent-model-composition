import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

class OxfordPets(Dataset):

    def __init__(self, root='/', train=True, transform=None, blurr_class=None, index_list=None):
        self.base_folder = os.path.join(root, 'OxfordPets')
        self.transform = transform
        self.train = train
        self.num_classes = 37
        split = 'train' if train else 'val'

        if split == 'train':
            path = os.path.join(self.base_folder, 'annotations', 'trainval.txt')
        else:
            path = os.path.join(self.base_folder, 'annotations', 'test.txt')

        with open(path) as f:
            data = [l.strip() for l in f.readlines()]

        self.targets = [int(l.split(' ')[1]) - 1 for l in data]
        self.samples = [os.path.join(self.base_folder, 'images', l.split(' ')[0] + '.jpg') for l in data]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target = self.targets[index]
        img = Image.open(self.samples[index])
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target
