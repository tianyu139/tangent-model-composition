import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import glob

class MIT67(Dataset):

    def __init__(self, root='/', train=True, transform=None):
        self.base_folder = os.path.join(root, 'MIT67')
        self.transform = transform
        self.train = train
        self.num_classes = 67
        split = 'train' if train else 'val'
        folder_targets = {os.path.basename(f[:-1]):i for i, f in enumerate(sorted(glob.glob(os.path.join(self.base_folder, 'Images/*/'))))}

        train_images_path = os.path.join(self.base_folder, 'TrainImages.txt')
        test_images_path = os.path.join(self.base_folder, 'TestImages.txt')

        if split == 'train':
            with open(train_images_path) as f:
                paths = f.readlines()
        else:
            with open(test_images_path) as f:
                paths = f.readlines()
        paths = [p.strip() for p in paths]
        self.samples = [os.path.join(self.base_folder, 'Images', p) for p in paths]
        self.targets = [folder_targets[p.split('/')[0]] for p in paths]

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

