from typing import List, Union

import PIL
import os
import copy
import numpy as np
import random

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset

IMAGENET_NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406],
                                          [0.229, 0.224, 0.225])
MNIST_NORMALIZE = transforms.Normalize((0.1307,), (0.3081,))
CIFAR_NORMALIZE = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


def get_dataset(dataset, root='./data', n_tasks=1, task_split=None):
    if dataset == 'mit67':
        from datasets.MIT67 import MIT67
        transform_train, transform_test = _get_imagenet_transforms()
        train_set = MIT67(root=root, train=True, transform=transform_train)
        test_set = MIT67(root=root, train=False, transform=transform_test)
    elif dataset == 'oxfordpets':
        from datasets.OxfordPets import OxfordPets
        transform_train, transform_test = _get_imagenet_transforms()
        train_set = OxfordPets(root=root, train=True, transform=transform_train)
        test_set = OxfordPets(root=root, train=False, transform=transform_test)
    else:
        raise ValueError("No such dataset")

    if n_tasks > 1:
        if task_split == 'class':
            train_set = ClassIncrementalDataset(train_set, n_tasks=n_tasks)
        elif task_split == 'rand':
            train_set = TaskIncrementalDataset(train_set, n_tasks=n_tasks)
        else:
            raise ValueError("No such task split")

    return train_set, test_set


def _get_imagenet_transforms(augment=True, input_size=224):
    resize = int(round(input_size * 256 / 224))
    transform_augment = transforms.Compose([
      transforms.Resize(resize),
      transforms.RandomCrop(input_size),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      IMAGENET_NORMALIZE
    ])
    transform_test = transforms.Compose([
      transforms.Resize(resize),
      transforms.CenterCrop(input_size),
      transforms.ToTensor(),
      IMAGENET_NORMALIZE
    ])
    transform_train = transform_augment if augment else transform_test
    return transform_train, transform_test


class ClassIncrementalDataset(Dataset):
    def __init__(self, dataset, n_tasks, transform=None):
        self.dataset = dataset
        self.n_classes_total = len(np.unique(dataset.targets))
        self.current_task = 0
        self.n_tasks = n_tasks
        self.task_n_classes = [self.n_classes_total // self.n_tasks for _ in range(n_tasks)]
        i = 0
        while sum(self.task_n_classes) != self.n_classes_total:
            self.task_n_classes[i] += 1
            i += 1
        self.transform = transform

        # Create fast way of finding task id
        _class_taskid = {}
        _taskid = 0
        _current_task_classes = 0
        for i in range(self.n_classes_total):
            if _current_task_classes == self.task_n_classes[_taskid]:
                _taskid += 1
                _current_task_classes = 0

            _class_taskid[i] = _taskid
            _current_task_classes += 1

        for i in range(self.n_tasks):
            assert len([x for x in _class_taskid.values() if x == i]) == self.task_n_classes[i]

        # Stores mapping of indices based on task id
        self.task_indices = {i:[] for i in range(n_tasks)}

        # Map index to task id
        for idx, label in enumerate(dataset.targets):
            task_id = _class_taskid[label]
            self.task_indices[task_id].append(idx)

        self.return_dataset_idx = False


    def __len__(self):
        return len(self.task_indices[self.current_task])

    def __getitem__(self, idx):
        dataset_idx = self.task_indices[self.current_task][idx]

        data, label = self.dataset[dataset_idx]

        if self.transform is not None:
            if self.return_dataset_idx:
                return self.transform(data), label, dataset_idx
            else:
                return self.transform(data), label
        else:
            if self.return_dataset_idx:
                return data, label, dataset_idx
            else:
                return data, label


class TaskIncrementalDataset(Dataset):
    def __init__(self, dataset, n_tasks, transform=None, imbalanced=False):
        self.dataset = dataset
        self.n_classes_total = len(np.unique(dataset.targets))
        self.current_task = 0
        self.n_tasks = n_tasks
        self.transform = transform

        # Stores mapping of indices based on task id
        self.task_indices = {i:[] for i in range(n_tasks)}

        # Ensure same shuffling each time
        rng = np.random.default_rng(139)

        shuffle_idxs = [i for i in range(len(dataset.targets))]
        rng.shuffle(shuffle_idxs)

        if imbalanced:
            steps = sum(range(1, n_tasks+1))
            step_size = len(dataset) / steps
            n_per_task = [int(step_size * i) for i in range(1, n_tasks+1)]
            i = n_tasks - 1
            while sum(n_per_task) != len(dataset):
                assert i > 0
                n_per_task[i] += 1
                i -= 1

            i = 0
            for task_id, n in enumerate(n_per_task):
                self.task_indices[task_id] += shuffle_idxs[i:i+n]
                i += n

            for i in range(n_tasks):
                assert len(self.task_indices[i]) == n_per_task[i]
        else:
            for i, idx in enumerate(shuffle_idxs):
                task_id = i % n_tasks
                self.task_indices[task_id].append(idx)

        self.return_dataset_idx = False

    def __len__(self):
        return len(self.task_indices[self.current_task])

    def __getitem__(self, idx):
        dataset_idx = self.task_indices[self.current_task][idx]

        data, label = self.dataset[dataset_idx]

        if self.transform is not None:
            if self.return_dataset_idx:
                return self.transform(data), label, dataset_idx
            else:
                return self.transform(data), label
        else:
            if self.return_dataset_idx:
                return data, label, dataset_idx
            else:
                return data, label

