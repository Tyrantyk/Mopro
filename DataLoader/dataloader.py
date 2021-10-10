from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import torch
import os
from PIL import ImageFile
import pickle
from noisy_dataset import noisify

ImageFile.LOAD_TRUNCATED_IMAGES = True



class cifar10_dataset(Dataset):
    def __init__(self, root_dir, transform, mode, num_class, transform_strong=None):
        self.root = root_dir
        self.transform = transform
        self.mode = mode

        if self.mode == 'test':
            data = self.load_file('./dataset/test_batch')
            self.val_imgs = np.transpose(np.reshape(data['data'], (10000, 3, 32, 32)), (0, 2, 3, 1))  # numpy 10000x3072
            self.val_labels = data['labels']  # list 10000

        else:
            self.transform_strong = transform_strong
            for i in range(5):
                data = self.load_file('./dataset/data_batch_' + str(i + 1))
                imgs = np.transpose(np.reshape(data['data'], (10000, 3, 32, 32)), (0, 2, 3, 1))# numpy 10000x3072
                labels = data['labels']  # list 10000

                if i == 0:
                    self.train_imgs = imgs
                    self.train_labels = labels
                else:
                    self.train_imgs = np.r_[self.train_imgs, imgs]
                    self.train_labels = self.train_labels + labels


    def __getitem__(self, index):
        if self.mode == 'train':
            target = self.train_labels[index]
            image = Image.fromarray(self.train_imgs[index])
            img = self.transform(image)
            img_aug = self.transform_strong(image)
            return img, target, img_aug

        elif self.mode == 'test':
            target = self.val_labels[index]
            image = Image.fromarray(self.val_imgs[index])
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_labels)
        else:
            return len(self.val_labels)

    def load_file(self,filename):
        with open(filename, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
            fo.close()
        return data


class cifar10_dataloader():
    def __init__(self, batch_size, num_class, num_workers, root_dir, distributed, crop_size=0.2):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.distributed = distributed

        self.transform_train = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(crop_size, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        self.transform_strong = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(crop_size, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        self.transform_test = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.494, 0.485, 0.450), (0.200, 0.199, 0.202)),
        ])

    def run(self):

        train_dataset = cifar10_dataset(root_dir=self.root_dir, transform=self.transform_train, mode="train",
                                          num_class=self.num_class, transform_strong=self.transform_strong)
        test_dataset = cifar10_dataset(root_dir=self.root_dir, transform=self.transform_test, mode='test',
                                         num_class=self.num_class)

        train_dataset = DatasetWrapper(train_dataset, 'symmetric', 0.4, noise_train=True, num_cls=10)

        if self.distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            eval_sampler = None
            test_sampler = None

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=True)

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=test_sampler)

        return train_loader, test_loader

class DatasetWrapper(torch.utils.data.Dataset):
    """Noise Dataset Wrapper"""

    def __init__(self, dataset, noise_type='clean', noise_rate=0,
                 yfile=None, weights_file=None, noise_train=False,
                 only_labeled=False, num_cls=10):
        """
        Args:
            dataset: the dataset to wrap, it should be an classification dataset
            noise_type: how to add noise for label: [clean/symmetric/asymmetric]
            noise_rate: noise ratio of adding noise
            yfile: The directory for the "y.npy" file. Once yfile assigned, we
                   will load yfile as labels and the given noise option will be
                   neglect. The weight of each sample will set to an binary
                   value according to the matching result of origin labels.
            weights_file: The weights for each samples, it should be an .npy
                   file of shape [len(dataset)] with either binary value or
                   probability value between [0,1]. "Specifically, all of the
                   unlabeled data should have zero-weight." The loaded weights
                   will multiply with the exists noise_or_not. So, it's ok to
                   give an weights for labeled data (noisy or clean).
        """
        self.dataset = dataset
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.num_cls = num_cls


        if noise_type == "clean":
            self.weights = np.ones(len(dataset))
            self.labels_to_use = dataset.targets
        else:
            # noisify labels
            train_clean_labels = np.expand_dims(np.asarray(dataset.train_labels), 1)
            train_noisy_labels, _ = noisify(train_labels=train_clean_labels,
                                            nb_classes=self.num_cls,
                                            noise_type=noise_type,
                                            noise_rate=noise_rate)
            self.labels_to_use = train_noisy_labels.flatten()
            assert len(self.labels_to_use) == len(dataset.train_labels)
            self.weights = (np.transpose(self.labels_to_use) ==
                            np.transpose(train_clean_labels)).squeeze()

        if noise_train:
            self.weights = np.ones(len(dataset))

        if weights_file is not None:
            # weights_file can be weights.npy or labeled.npy
            assert self.noise_type in ['preload', 'clean']
            self.useit = np.load(weights_file)
            assert len(self.useit) == len(dataset)
            if self.useit.dtype == np.bool:
                self.useit = self.useit.astype(np.float)
            self.weights = self.weights * self.useit


    def save_noise_labels(self, dir):
        np.save(dir, np.asarray(self.labels_to_use))

    def __getitem__(self, index):
        # self.noise_or_not can expand to the weights of sample. So we can load
        # Semi-Supervised dataset here.
        img, target_gt, img_aug = self.dataset[index]
        target_use = self.labels_to_use[index]
        weights = self.weights[index]
        return img, target_use, img_aug

    def __len__(self):
        return len(self.dataset)
