import os
import sys
import re
import datetime

import numpy
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import random
import pandas as pd
from collections import defaultdict
from scipy.stats.mstats import pearsonr, kendalltau, spearmanr


class EarlyStop:
    def __init__(self, maxmize=True, early_stop_epoch=5):
        self.decline_epoch = 0
        self.early_stop_epoch = early_stop_epoch
        self.maxmize = maxmize
        if maxmize:
            self.metric = 0
        else:
            self.metric = 10000

    def __call__(self, metric):
        if (self.metric < metric and self.maxmize) or (self.metric > metric and not self.maxmize):
            self.metric = metric
            self.decline_epoch = 0
        else:
            self.decline_epoch += 1

        if self.decline_epoch == self.early_stop_epoch:
            return True
        else:
            return False


def get_cifar100(dataset):
    if dataset == 'cifar100_train':
        return CIFAR100(root='~/cv_data/cifar100/data', train=True, download=False)
    elif dataset == 'cifar100_test':
        return CIFAR100(root='~/cv_data/cifar100/data', train=False, download=False)
    else:
        raise NotImplementedError


def get_diamond_class_info():
    df = pd.read_csv('data/sample_class_diamond.csv')
    return df

def get_titanic_class_info():
    df = pd.read_csv('data/titanic_sample_class.csv')
    return df


class CIFAR100WithIdx(CIFAR100):
    """
    Extends CIFAR100 dataset to yield index of element in addition to image and target label.
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(CIFAR100WithIdx, self).__init__(root=root,
                                              train=train,
                                              transform=transform,
                                              target_transform=target_transform,
                                              download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int):  index of element to be fetched

        Returns:
            tuple: (index， sample, target) where index is the index of this sample in dataset.
        """
        img, target = super().__getitem__(index)
        return index, img, target


class CIFAR10WithIdx(CIFAR10):
    """
    Extends CIFAR100 dataset to yield index of element in addition to image and target label.
    """

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super(CIFAR10WithIdx, self).__init__(root=root,
                                              train=train,
                                              transform=transform,
                                              target_transform=target_transform,
                                              download=download)

    def __getitem__(self, index):
        """
        Args:
            index (int):  index of element to be fetched

        Returns:
            tuple: (index， sample, target) where index is the index of this sample in dataset.
        """
        img, target = super().__getitem__(index)
        return index, img, target


def get_cv_generator(mean, std, batch_size=16, dataset='cifar100', num_workers=1, shuffle=True):
    """ return cross validation generator
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cv_generator:train and valid dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    kfold = KFold(n_splits=5, shuffle=True, random_state=2021)

    if dataset == 'cifar100':
        cifar_training = CIFAR100WithIdx(root='~/cv_data/cifar100/data', train=True, download=False, transform=transform_train)
    elif dataset == 'cifar10':
        cifar_training = CIFAR10WithIdx(root='~/cv_data/cifar100/data', train=True, download=False, transform=transform_train)

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(cifar_training)):
        print(f'FOLD {fold}')
        print('----------------------------')
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(cifar_training, batch_size=batch_size, sampler=train_subsampler, num_workers=num_workers)
        validloader = torch.utils.data.DataLoader(cifar_training, batch_size=batch_size, sampler=valid_subsampler, num_workers=num_workers)

        # check
        # for train_ids, images, labels in trainloader:
        #     assert numpy.alltrue(numpy.isin(train_ids.numpy(), train_ids))

        yield trainloader, validloader

    # cifar100_training_loader = DataLoader(
    #     cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    # return cifar100_training_loader


def balanced_sampling_cifar_data(sample_num_each_class):
    cifar100_training = get_cifar100('cifar100_train')
    targets = cifar100_training.targets
    class2sample_ids = defaultdict(list)
    for i, target in enumerate(targets):
        class2sample_ids[target].append(i)
    
    sample_ids = []
    for image_class in class2sample_ids:
        sample_ids_class = class2sample_ids[image_class]
        random.shuffle(sample_ids_class)
        sample_ids.extend(random.sample(sample_ids_class, k=sample_num_each_class))
    
    return sample_ids


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_cv_generator_balance_class(mean, std, batch_size=16, dataset='cifar100', num_workers=1, shuffle=True):
    """ return cross validation generator
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cv_generator:train and valid dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if dataset == 'cifar100':
        cifar_training = CIFAR100WithIdx(root='~/cv_data/cifar100/data', train=True, download=False, transform=transform_train)
    elif dataset == 'cifar10':
        cifar_training = CIFAR10WithIdx(root='~/cv_data/cifar100/data', train=True, download=False, transform=transform_train)

    sample_label2sample_id = defaultdict(list)
    for i, sample_label in enumerate(cifar_training.targets):
        sample_label2sample_id[sample_label].append(i)
    
    for sample_label in sample_label2sample_id:
        random.shuffle(sample_label2sample_id[sample_label])

    Fold_K = 5
    for fold in range(Fold_K):
        print(f'FOLD {fold}')
        print('----------------------------')
        train_ids, valid_ids = [], []
        for sample_ids in sample_label2sample_id.values():
            for i in range(len(sample_ids)):
                if i % Fold_K == fold:
                    valid_ids.append(sample_ids[i])
                else:
                    train_ids.append(sample_ids[i])
        
        assert len(train_ids) == (Fold_K - 1) * len(valid_ids)
        assert len(set(train_ids) & set(valid_ids)) == 0
        random.shuffle(train_ids)
        random.shuffle(valid_ids)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)

        train_loader = torch.utils.data.DataLoader(cifar_training, batch_size=batch_size, num_workers=num_workers,
                                                sampler=train_subsampler)
        valid_loader = torch.utils.data.DataLoader(cifar_training, batch_size=batch_size, num_workers=num_workers,
                                                sampler=valid_subsampler)
        yield train_loader, valid_loader


def get_train_dataloader(mean, std, batch_size=16, dataset='cifar100', num_workers=1, shuffle=True, transform=None):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    if not transform:
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform_train = transform

    if dataset == 'cifar100':
        cifar100_training = CIFAR100WithIdx(root='~/cv_data/cifar100/data', train=True, download=False, transform=transform_train)
    else:
        cifar100_training = CIFAR10WithIdx(root='~/cv_data/cifar100/data', train=True, download=False, transform=transform_train)

    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=16, dataset='cifar100', num_workers=1, shuffle=True, transform=None):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if not transform:
        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    else:
        transform_test = transform

    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    if dataset == 'cifar100':
        cifar100_test = CIFAR100WithIdx(root='~/cv_data/cifar100/data', train=False, download=False, transform=transform_test)
    else:
        cifar100_test = CIFAR10WithIdx(root='~/cv_data/cifar100/data', train=False, download=False, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]


def save_result_to_excel(df_describe, path):
    mean_std_column = ['mae', 'rmse', 'auc', 'acc', 'f1', 'pearson_cor', 'spearman_cor', 'kendall_cor']
    max_column = ['pearson_p_value', 'spearman_p_value', 'kendall_p_values']

    new_df = pd.DataFrame()
    for column in df_describe.columns:
        if column in mean_std_column:
            new_df[f'{column}_mean'] = [round(df_describe.loc['mean', column], 5)]
            new_df[f'{column}_std'] = [round(df_describe.loc['std', column], 5)]
        elif column in max_column:
            new_df[f'{column}_max'] = [df_describe.loc['max', column]]
        else:
            new_df[f'{column}_mean'] = [round(df_describe.loc['mean', column], 5)]
            new_df[f'{column}_std'] = [round(df_describe.loc['std', column], 5)]
    
    # new_df.to_excel(path)
    # df_describe.to_excel(path.replace('.xlsx', '_all.xlsx'))
    if 'mae' in df_describe.columns:
        return df_describe.loc['mean', 'mae']
    else:
        return df_describe.loc['mean', 'auc']


def correlation(x, y):
    pearson_cor, pearson_p_value = pearsonr(x, y)
    spearman_cor, spearman_p_value = spearmanr(x, y)
    kendall_cor, kendall_p_value = kendalltau(x, y)

    print(f'pearson_cor: {pearson_cor}, pearson_p_value: {pearson_p_value}')
    print(f'spearman_cor: {spearman_cor}, spearman_p_value: {spearman_p_value}')
    print(f'kendall_cor: {kendall_cor}, kendall_p_value: {kendall_p_value}')

    return pearson_cor, pearson_p_value, spearman_cor, spearman_p_value, kendall_cor, kendall_p_value


if __name__ == '__main__':
    cifar100_training = CIFAR100WithIdx(root='~/cv_data/cifar100/data', train=True, download=False)
    print('yes')
