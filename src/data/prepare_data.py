from torch.utils.data import DataLoader
from PIL import Image
from torchvision import datasets, transforms
from collections import OrderedDict
import torch.utils.data as data
import numpy as np
import pickle
import os
import lmdb
import six
import json

from .const import GTSRB_LABEL_MAP


def loads_data(buf):
    return pickle.loads(buf)

class LMDBDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super().__init__()
        db_path = os.path.join(root, f"{split}.lmdb")
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf)
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class COOPLMDBDataset(LMDBDataset):
    def __init__(self, root, split="train", transform=None) -> None:
        super().__init__(root, split, transform=transform)
        with open(os.path.join(root, "split.json")) as f:
            split_file = json.load(f)
        idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split_file["test"]}.items()))
        self.classes = list(idx_to_class.values())

def refine_classnames(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names

def prepare_padding_data(dataset, data_path):
    data_path = os.path.join(data_path, dataset)
    if dataset == "cifar10":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR10(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "cifar100":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR100(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "gtsrb":
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(list(GTSRB_LABEL_MAP.values())),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "svhn":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.SVHN(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': [f'{i}' for i in range(10)],
            'mask': np.zeros((32, 32)),
        }
    elif dataset in ["food101", "eurosat", "sun397", "ucf101", "stanfordcars", "flowers102"]:
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=8),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")
    return loaders, configs


def prepare_watermarking_data(dataset, data_path, preprocess, test_process=None, shuffle=True, batch_size=None):
    data_path = os.path.join(data_path, dataset)
    if not test_process:
        test_process = preprocess
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = False, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        batch_size = batch_size if batch_size else 256
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = False, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        batch_size = batch_size if batch_size else 256
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset == "svhn":
        train_data = datasets.SVHN(root = data_path, split="train", download = False, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = False, transform = test_process)
        class_names = [f'{i}' for i in range(10)]
        batch_size = batch_size if batch_size else 256
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset in ["food101", "sun397", "eurosat", "ucf101", "stanfordcars", "flowers102"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = test_process)
        class_names = refine_classnames(test_data.classes)
        batch_size = batch_size if batch_size else 256
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=8),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = test_process)
        class_names = refine_classnames(test_data.classes)
        batch_size = batch_size if batch_size else 64
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
    elif dataset == "gtsrb":
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = test_process)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        batch_size = batch_size if batch_size else 256
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")

    print(f'Dataset: {dataset}')
    print(f'Train data size: {len(train_data)}')
    print(f'Test data size: {len(test_data)}')
    print(f'Batch size: {batch_size}')
    return loaders, class_names

def prepare_plain_data(dataset, data_path, preprocess, test_process=None, shuffle=True, batch_size=None):
    data_path = os.path.join(data_path, dataset)
    if not test_process:
        test_process = preprocess
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = False, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = False, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset == "svhn":
        train_data = datasets.SVHN(root = data_path, split="train", download = False, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = False, transform = test_process)
        class_names = [f'{i}' for i in range(10)]
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset in ["food101", "sun397", "eurosat", "ucf101", "stanfordcars", "flowers102"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = test_process)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=8),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = test_process)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
    elif dataset == "gtsrb":
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = test_process)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")

    return loaders, class_names

from torch.utils.data import Subset
import random

def sample_subset(dataset, ratio=0.5, seed=None):
    assert 0 < ratio <= 1.0, "Ratio must be between 0 and 1"
    length = len(dataset)
    n_samples = int(length * ratio)

    indices = list(range(length))
    if seed is not None:
        random.seed(seed)
    random.shuffle(indices)

    subset_indices = indices[:n_samples]
    return Subset(dataset, subset_indices)



def prepare_plain_data_sub(dataset, data_path, preprocess, ratio=0.1, test_process=None, shuffle=True, seed=42):
    data_path = os.path.join(data_path, dataset)
    if not test_process:
        test_process = preprocess
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = False, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = False, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset == "svhn":
        train_data = datasets.SVHN(root = data_path, split="train", download = False, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = False, transform = test_process)
        class_names = [f'{i}' for i in range(10)]
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset in ["food101", "sun397", "eurosat", "ucf101", "stanfordcars", "flowers102"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = test_process)
        print(f"Length of train data: {len(train_data)}")
        train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        print(f"Length of sampled train data: {len(train_data)}")
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=8),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = test_process)
        print(f"Length of train data: {len(train_data)}")
        train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        print(f"Length of sampled train data: {len(train_data)}")
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
    elif dataset == "gtsrb":
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = test_process)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")

    return loaders, class_names

import random
from torch.utils.data import Subset
import os

def sample_fixed_number_per_class(dataset, num_samples_per_class, seed=None):
    if seed is not None:
        random.seed(seed)
    
    # We need a way to fetch labels; assuming it's a property or can be accessed similarly
    # Since not shown how labels are accessed directly, here's a general approach:
    class_to_indices = {}
    for i in range(len(dataset)):
        # Assuming the second element of the tuple returned by __getitem__ is the label
        _, label = dataset[i]  
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(i)

    subset_indices = []
    for indices in class_to_indices.values():
        if len(indices) >= num_samples_per_class:
            subset_indices.extend(random.sample(indices, num_samples_per_class))
        else:
            # If not enough samples in class, take all available
            subset_indices.extend(indices)

    return Subset(dataset, subset_indices)


def prepare_plain_data_few_shot(dataset, data_path, preprocess, num_samples_per_class=10, test_process=None, shuffle=True, seed=42):
    data_path = os.path.join(data_path, dataset)
    if not test_process:
        test_process = preprocess
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = False, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = False, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset == "svhn":
        train_data = datasets.SVHN(root = data_path, split="train", download = False, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = False, transform = test_process)
        class_names = [f'{i}' for i in range(10)]
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset in ["food101", "sun397", "eurosat", "ucf101", "stanfordcars", "flowers102"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = test_process)
        class_names = refine_classnames(test_data.classes)
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=8),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = test_process)
        class_names = refine_classnames(test_data.classes)
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
    elif dataset == "gtsrb":
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = test_process)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")

    return loaders, class_names


def prepare_watermarking_data_few_shot(dataset, data_path, preprocess, test_process=None, shuffle=True, batch_size=None, num_samples_per_class=16, seed=42):
    data_path = os.path.join(data_path, dataset)
    if not test_process:
        test_process = preprocess
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = False, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        batch_size = batch_size if batch_size else 256
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = False, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        batch_size = batch_size if batch_size else 256
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset == "svhn":
        train_data = datasets.SVHN(root = data_path, split="train", download = False, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = False, transform = test_process)
        class_names = [f'{i}' for i in range(10)]
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        batch_size = batch_size if batch_size else 256
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    elif dataset in ["food101", "sun397", "eurosat", "ucf101", "stanfordcars", "flowers102"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = test_process)
        class_names = refine_classnames(test_data.classes)
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        batch_size = batch_size if batch_size else 256
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=8),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = test_process)
        class_names = refine_classnames(test_data.classes)
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        batch_size = batch_size if batch_size else 64
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
    elif dataset == "gtsrb":
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = test_process)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        batch_size = batch_size if batch_size else 256
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")

    print(f'Dataset: {dataset}')
    print(f'Train data size: {len(train_data)}')
    print(f'Test data size: {len(test_data)}')
    print(f'Batch size: {batch_size}')
    return loaders, class_names


import torch
from torch.utils.data import Dataset, DataLoader

def single_round_API_inference(loaders, network, device, batch_size=64, num_workers=4):
    """
    Run a forward pass using `network` on both train and test data,
    and return new DataLoaders with (input, label, logits) for each split.

    Args:
        loaders (dict): Dictionary with 'train' and 'test' DataLoaders.
        network (function): Function taking a batch of inputs and returning logits.
        device (torch.device): Device for inference.
        batch_size (int): Batch size for returned DataLoaders.
        num_workers (int): Number of workers for returned DataLoaders.

    Returns:
        dict: New DataLoaders with keys 'train' and 'test'.
    """

    class TripletDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    output_loaders = {}

    for split in ['train', 'test']:
        data = []
        loader = loaders[split]
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = network(inputs)
                data.extend([
                    (inp.cpu(), lbl.cpu(), log.cpu())
                    for inp, lbl, log in zip(inputs, labels, logits)
                ])

        print(f"Processed {len(data)} samples for {split} split.")
        dataset = TripletDataset(data)
        output_loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == 'train'), num_workers=num_workers
        )

    return output_loaders

def prepare_padding_data_few_shot(dataset, data_path, num_samples_per_class=16, seed=42, mask_size=None):
    data_path = os.path.join(data_path, dataset)
    if dataset == "cifar10":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR10(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = True, transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "cifar100":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR100(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = True, transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "gtsrb":
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(list(GTSRB_LABEL_MAP.values())),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "svhn":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.SVHN(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = True, transform = preprocess)
        class_names = [f'{i}' for i in range(10)]
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': [f'{i}' for i in range(10)],
            'mask': np.zeros((32, 32)),
        }
    elif dataset in ["food101", "eurosat", "sun397", "ucf101", "stanfordcars", "flowers102"]:
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            # transforms.Resize((180, 180)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        loaders = {
            'train': DataLoader(train_data, 256, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 256, shuffle = False, num_workers=8),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
            # 'mask': np.zeros((180, 180)),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            # transforms.Resize((200, 200)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        print(f"Length of train data: {len(train_data)}")
        # train_data = sample_subset(train_data, ratio=ratio, seed=seed)
        train_data = sample_fixed_number_per_class(train_data, num_samples_per_class, seed=seed)
        print(f"Length of sampled train data: {len(train_data)} = {num_samples_per_class * len(class_names)}")
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
        print(200,200)
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
            # 'mask': np.zeros((200, 200)),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")
    return loaders, configs

   