import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from data.transform import train_transform_cifar, train_transform_imagenet, query_transform, encode_onehot, Onehot


def load_data(dataset, root, batch_size, workers):
    """
    Load imagenet dataset

    Args
        root (str): Path of imagenet dataset.
        batch_size (int): Number of samples in one batch.
        workers (int): Number of data loading threads.

    Returns
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
    """

    # Construct data loader
    sub = dataset.split('-')[2]
    dataset_name = dataset.split('-')[0]

    if dataset_name == 'cifar':
        if sub == 'IF100':
            train_dir = os.path.join(root, 'cifar-100-IF100', 'images', 'train')
            new_root = os.path.join(root, 'cifar-100-IF100')
        elif sub == 'IF50':
            train_dir = os.path.join(root, 'cifar-100-IF50', 'images', 'train')
            new_root = os.path.join(root, 'cifar-100-IF50')
        elif sub == 'IF20':
            train_dir = os.path.join(root, 'cifar-100-IF20', 'images', 'train')
            new_root = os.path.join(root, 'cifar-100-IF20')
        elif sub == 'IF10':
            train_dir = os.path.join(root, 'cifar-100-IF10', 'images', 'train')
            new_root = os.path.join(root, 'cifar-100-IF10')
        elif sub == 'IF1':
            train_dir = os.path.join(root, 'cifar-100-IF1', 'images', 'train')
            new_root = os.path.join(root, 'cifar-100-IF1')
        else:
            print('train path error')
            return
        query_dir = os.path.join(new_root, 'images', 'query')
        database_dir = os.path.join(new_root, 'images', 'database')

    elif dataset_name == 'imagenet':
        if sub == 'IF100':
            train_dir = os.path.join(root, 'train-alpha=0.99-IF=100.0')
        elif sub == 'IF50':
            train_dir = os.path.join(root, 'train-alpha=0.845-IF=50.0')
        elif sub == 'IF20':
            train_dir = os.path.join(root, 'train-IF20')
        elif sub == 'IF10':
            train_dir = os.path.join(root, 'train-IF10')
        elif sub == 'IF1':
            train_dir = os.path.join(root, 'train')
        else:
            print('train path error')
            return

        query_dir = os.path.join(root, 'query')
        database_dir = os.path.join(root, 'database')
    else:
        print('train path error')
        return

    if dataset_name == 'cifar':

        train_data_loader = DataLoader(
            ImagenetDataset(
                train_dir,
                transform=train_transform_cifar(),
                targets_transform=Onehot(100),
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )
    else:
        train_data_loader = DataLoader(
            ImagenetDataset(
                train_dir,
                transform=train_transform_imagenet(),
                targets_transform=Onehot(100),
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
        )

    query_data_loader = DataLoader(
        ImagenetDataset(
            query_dir,
            transform=query_transform(),
            targets_transform=Onehot(100),
        ),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
    )

    database_data_loader = DataLoader(
        ImagenetDataset(
            database_dir,
            transform=query_transform(),
            targets_transform=Onehot(100),
        ),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
    )

    return train_data_loader, query_data_loader, database_data_loader


class ImagenetDataset(Dataset):
    classes = None
    class_to_idx = None

    def __init__(self, root, transform=None, targets_transform=None):
        self.root = root
        self.transform = transform
        self.targets_transform = targets_transform
        self.data = []
        self.targets = []

        # Assume file alphabet order is the class order
        if ImagenetDataset.class_to_idx is None:
            ImagenetDataset.classes, ImagenetDataset.class_to_idx = self._find_classes(root)

        for i, cl in enumerate(ImagenetDataset.classes):
            cur_class = os.path.join(self.root, cl)
            files = os.listdir(cur_class)
            files = [os.path.join(cur_class, i) for i in files]
            self.data.extend(files)
            self.targets.extend([ImagenetDataset.class_to_idx[cl] for i in range(len(files))])
        self.targets = np.asarray(self.targets)
        self.onehot_targets = torch.from_numpy(encode_onehot(self.targets, 100)).float()

    def get_onehot_targets(self):
        return self.onehot_targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.targets_transform is not None:
            target = self.targets_transform(target)
        return img, target, item

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
