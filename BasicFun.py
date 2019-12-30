import logging
import colorlog
import math
import os
import pickle
from inspect import stack
from math import factorial

import cv2
import numpy as np
import torch as tc
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from termcolor import cprint
from torch import nn


def choose_device(n=0):
    return tc.device("cuda:"+str(n) if tc.cuda.is_available() else "cpu")


def project_path(project='T-Nalg/'):
    cur_path = os.path.abspath(os.path.dirname(__file__))
    return cur_path[:cur_path.find(project) + len(project)]


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    path_flag = os.path.exists(path)
    if not path_flag:
        os.makedirs(path)
    return path_flag


def save(path, file, data, names):
    mkdir(path)
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    tc.save(tmp, os.path.join(path, file))


def save_pr(path, file, data, names):
    mkdir(path)
    # print(os.path.join(path, file))
    s = open(os.path.join(path, file), 'wb')
    tmp = dict()
    for i in range(0, len(names)):
        tmp[names[i]] = data[i]
    pickle.dump(tmp, s)
    s.close()


def load(path_file, names=None, device='cpu'):
    if os.path.isfile(path_file):
        if names is None:
            data = tc.load(path_file)
            return data
        else:
            tmp = tc.load(path_file, map_location=device)
            if type(names) is str:
                data = tmp[names]
                return data
            elif (type(names) is list) or (type(names) is tuple):
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
                return tuple(data)
    else:
        return False


def load_pr(path_file, names=None):
    if os.path.isfile(path_file):
        s = open(path_file, 'rb')
        if names is None:
            data = pickle.load(s)
            s.close()
            return data
        else:
            tmp = pickle.load(s)
            if type(names) is str:
                data = tmp[names]
                s.close()
                return data
            elif (type(names) is list) or (type(names) is tuple):
                nn = len(names)
                data = list(range(0, nn))
                for i in range(0, nn):
                    data[i] = tmp[names[i]]
                s.close()
                return tuple(data)
    else:
        return False


def output_txt(x, filename='data'):
    np.savetxt(filename + '.txt', x)


def print_dict(a, keys=None, welcome='', style_sep=': ', color='white', end='\n'):
    express = welcome
    if keys is None:
        for n in a:
            express += n + style_sep + str(a[n]) + end
    else:
        if type(keys) is str:
            express += keys.capitalize() + style_sep + str(a[keys])
        else:
            for n in keys:
                express += n.capitalize() + style_sep + str(a[n])
                if n is not keys[-1]:
                    express += end
    cprint(express, color)
    return express


def load_mnist_and_batch(dataset='MNIST', classes=None, num_s=None, num_t=None,
                         batch_size=100, data_dim=2, labels2mat=False, channel=None,
                         train_or_test=None, project_name=None, dev=None,
                         dtype=None):
    """
    data_dim: 5 means (samples, channels, d, size_x, size_y)
              2 means (samples, numel)
    """
    if (classes is None) or (classes is 'all'):
        classes = list(range(10))
    if num_s is None:
        num_s = ['all'] * classes.__len__()
    if num_t is None:
        num_t = ['all'] * classes.__len__()
    if dtype is None:
        dtype = tc.float32
    if dataset.lower() not in ['mnist', 'fashionmnist', 'fashion-mnist']:
        warning('This dataset [' + dataset + '] cannot be loaded by this function')
    flag_train = (len(classes) == 10)
    flag_test = (len(classes) == 10)
    for n in num_s:
        flag_train = (flag_train and (n == 'all'))
    for n in num_t:
        flag_test = (flag_test and (n == 'all'))

    samples, labels, samples_t, labels_t = \
        auto_load_dataset(dataset=dataset, project_name=project_name,
                          train_or_test=train_or_test, dev=dev)

    if (not flag_train) and (train_or_test in ['train', 'both', None]):
        samples, labels = select_samples(samples, labels, classes, num_s)
    if (not flag_test) and (train_or_test in ['test', 'both', None]):
        samples_t, labels_t = select_samples(samples_t, labels_t, classes, num_t)
    train_set, test_set = None, None
    if train_or_test in ['train', 'both', None]:
        train_set = batch_dataset(samples.type(dtype), labels, batch_size,
                                  data_dim, labels2mat, channel)
    if train_or_test in ['test', 'both', None]:
        test_set = batch_dataset(samples_t.type(dtype), labels_t, batch_size,
                                 data_dim, labels2mat, channel)
    return train_set, test_set


def load_cifar10_and_batch(classes=None, num_s=None, num_t=None,
                           batch_size=100, data_dim=4, labels2mat=False, channel=None,
                           train_or_test=None, project_name=None, dev=None, dtype=None):
    """
    To be fixed
    """
    dataset = 'cifar10'
    if num_s is None:
        num_s = ['all'] * classes.__len__()
    if num_t is None:
        num_t = ['all'] * classes.__len__()
    if dtype is None:
        dtype = tc.float32

    samples, labels, samples_t, labels_t = \
        auto_load_dataset(dataset=dataset, project_name=project_name,
                          train_or_test=train_or_test, dev=dev)
    samples, labels = select_samples(samples, labels, classes, num_s)
    samples_t, labels_t = select_samples(samples_t, labels_t, classes, num_t)
    samples = samples.permute(0, 3, 1, 2)
    samples_t = samples_t.permute(0, 3, 1, 2)
    train_set = batch_dataset(samples.type(dtype), labels, batch_size,
                              data_dim, labels2mat, channel)
    test_set = batch_dataset(samples_t.type(dtype), labels_t, batch_size,
                             data_dim, labels2mat, channel)
    return train_set, test_set


def auto_load_dataset(dataset, project_name, train_or_test, dev):
    """
    data_dim: 4 means (samples, channels, size_x, size_y)
              2 means (samples, numel)
    """
    dataset = dataset.lower()
    if project_name is None:
        path_proj = '../'
    else:
        path_proj = project_path(project=project_name)
    if dev is None:
        dev = choose_device()

    path_proj = os.path.join(path_proj, '../Datasets')
    transform = transforms.ToTensor()

    trainset, testset = None, None
    case = None
    if dataset == 'mnist':
        if train_or_test in ['train', 'both', None]:
            trainset = tv.datasets.MNIST(
                root=path_proj,
                train=True,
                download=True,
                transform=transform)
        if train_or_test in ['test', 'both', None]:
            testset = tv.datasets.MNIST(
                root=path_proj,
                train=False,
                download=True,
                transform=transform)
        case = 1
    elif (dataset == 'fashionmnist') or \
            (dataset == 'fashion-mnist') or (dataset == 'fashion_mnist'):
        if train_or_test in ['train', 'both', None]:
            trainset = tv.datasets.FashionMNIST(
                root=path_proj,
                train=True,
                download=True,
                transform=transform)
        if train_or_test in ['test', 'both', None]:
            testset = tv.datasets.FashionMNIST(
                root=path_proj,
                train=False,
                download=True,
                transform=transform)
        case = 1
    elif (dataset == 'cifar10') or \
            (dataset == 'cifar-10'):
        if train_or_test in ['train', 'both', None]:
            trainset = tv.datasets.CIFAR10(
                root=path_proj,
                train=True,
                download=True,
                transform=transform)
        if train_or_test in ['test', 'both', None]:
            testset = tv.datasets.CIFAR10(
                root=path_proj,
                train=False,
                download=True,
                transform=transform)
        case = 2

    samples, labels = None, None
    samples_t, labels_t = None, None
    if case == 1:
        if train_or_test in ['train', 'both', None]:
            samples = trainset.train_data
            labels = trainset.train_labels
        if train_or_test in ['test', 'both', None]:
            samples_t = testset.test_data
            labels_t = testset.test_labels
    elif case == 2:
        if train_or_test in ['train', 'both', None]:
            samples = tc.from_numpy(trainset.data)
            labels = tc.from_numpy(np.array(trainset.targets))
        if train_or_test in ['test', 'both', None]:
            samples_t = tc.from_numpy(testset.data)
            labels_t = tc.from_numpy(np.array(testset.targets))
    else:
        warning('Something wrong in function \'auto_load_dataset\'')
        # samples, labels, samples_t, labels_t = None, None, None, None
    if train_or_test in ['train', 'both', None]:
        samples, labels = samples.to(dev), labels.to(dev)
    if train_or_test in ['test', 'both', None]:
        samples_t, labels_t = samples_t.to(dev), labels_t.to(dev)
    return samples, labels, samples_t, labels_t


def select_samples(samples, labels, classes, nums):

    def _select_samples(data, label, which):
        if data.ndimension() == 2:  # num_s * features
            data1 = data[which, :]
        elif data.ndimension() == 3:  # num_s * size_x * size_y
            data1 = data[which, :, :]
        else:    # num_s * size_x * size_y * RGB_channels
            data1 = data[which, :, :, :]
        return data1, label[which]

    def random_select_samples(data, label, num):
        which = tc.randperm(label.numel())[:min(num, label.numel())]
        data1, label1 = _select_samples(data, label, which)
        return data1, label1

    if type(classes) is int:  # select one class
        samples, labels = _select_samples(samples, labels, labels == classes)
        if (type(nums) is list) or (type(nums) is tuple):
            nums = nums[0]
        if nums != 'all':
            samples, labels = random_select_samples(samples, labels, nums)
    else:  # select multiple classes
        selected_samples = [None] * len(classes)
        selected_labels = [None] * len(classes)
        for n in range(len(classes)):
            selected_samples[n], selected_labels[n] = \
                _select_samples(samples, labels, labels == classes[n])
            if nums[n] != 'all':
                selected_samples[n], selected_labels[n] = \
                    random_select_samples(selected_samples[n], selected_labels[n], nums[n])
        samples = tc.cat(selected_samples, dim=0)
        labels = tc.cat(selected_labels)
    for n in range(len(classes)):
        labels[labels == classes[n]] = n
    return samples, labels


def batch_dataset(samples, labels, batch_size, data_dim, labels2mat,
                  channel=None, normalize=255):
    if (batch_size == 'all') or (batch_size is None):
        if labels2mat:
            labels = labels_to_matrix(labels, channel)
        size = samples.shape
        if data_dim == 2 and samples.ndimension() == 3:
            tmp = samples.reshape(size[0], size[1] * size[2])
        elif data_dim == 4 and samples.ndimension() == 3:
            tmp = samples.reshape(size[0], 1, size[1], size[2])
        elif data_dim == 3 and samples.ndimension() == 4:  # cifar10 before feature map
            tmp = samples.reshape(size[0], size[1], size[2] * size[3])
        elif data_dim == 5 and samples.ndimension() == 3:
            tmp = samples.reshape(size[0], 1, 1, size[1], size[2])
        elif data_dim == 5 and samples.ndimension() == 4:  # cifar10 before feature map
            tmp = samples.reshape(size[0], size[1], 1, size[2], size[3])
        else:  # data_dim.__len__() == samples.ndimension()
            tmp = samples
        batched_data = [(tmp/normalize, labels)]
    else:
        rand = tc.randperm(labels.shape[0])
        num_b = (labels.shape[0] // batch_size) + ((labels.shape[0] % batch_size) > 0)
        batched_data = list()
        for n in range(num_b):
            x = rand[n * batch_size: min((n + 1) * batch_size, labels.shape[0])]
            if data_dim == 2 and samples.ndimension() == 3:
                tmp = samples[x, :, :].reshape(
                    x.shape[0], samples.shape[1] * samples.shape[2])
            elif data_dim == 4 and samples.ndimension() == 3:
                tmp = samples[x, :, :].reshape(
                    x.shape[0], 1, samples.shape[1], samples.shape[2])
            elif data_dim == 3 and samples.ndimension() == 4:  # cifar10 before feature map
                tmp = samples[x, :, :, :].reshape(
                    x.shape[0], samples.shape[1], samples.shape[2]*samples.shape[3])
            elif data_dim == 5 and samples.ndimension() == 3:
                tmp = samples[x, :, :].reshape(
                    (x.shape[0], 1, 1) + samples.shape[1:])
            elif data_dim == 5 and samples.ndimension() == 4:  # cifar10 before feature map
                tmp = samples[x, :, :, :].reshape(
                    (x.shape[0], samples.shape[1], 1) + samples.shape[2:])
            else:  # data_dim == samples.ndimension()
                exp = 'samples[x'
                exp += (', :' * (samples.ndimension()-1)) + ']'
                tmp = eval(exp)
            tmp_l = labels[x]
            if labels2mat:
                tmp_l = labels_to_matrix(tmp_l, channel)
            batched_data.append((tmp/normalize, tmp_l))
    return batched_data


def resize_loader_mnist(data_loader, size0, size1):
    # use this function before feature map
    if (size0[0] == size1[0]) and (size0[1] == size1[1]):
        return data_loader
    else:
        s = data_loader[0][0].shape
        dim1 = np.prod(s[1:])
        if data_loader[0][0].ndimension() == 4:
            s1 = [s[1], size1[0], size1[1]]
        elif data_loader[0][0].ndimension() == 2:
            s1 = [size1[0] * size1[1]]
        else:
            s1 = [size1[0], size1[1]]
        device = data_loader[0][0].device
        dtype = data_loader[0][0].dtype
        data_loader1 = list()
        for img, labels in data_loader:
            img = img.reshape(img.shape[0], dim1)
            tmp = tc.zeros((img.shape[0], size1[0] * size1[1]), device=device, dtype=dtype)
            for n in range(img.shape[0]):
                _tmp = img[n, :].reshape(size0).cpu().numpy()
                _tmp = cv2.resize(_tmp, tuple(size1)).reshape(-1, )
                tmp[n, :] = tc.from_numpy(_tmp).to(device)
            data_loader1.append((tmp.reshape([img.shape[0]] + s1), labels))
        return data_loader1


def binary_loader(data_loader):
    data_loader1 = list()
    for img, labels in data_loader:
        img[img > 0.5] = 1
        img[img <= 0.5] = 0
        data_loader1.append((img, labels))
    return data_loader1


def resize_loader_cifar10(data_loader, size):
    # use this function before feature map
    size0 = (3, 32, 32)
    s = data_loader[0][0].shape
    dim1 = np.prod(s[2:])
    if data_loader[0][0].ndimension() == 4:
        s1 = [s[1], size[0], size[1]]
    else:  # data_loader[0][0].ndimension() == 3
        s1 = [s[1], size[0] * size[1]]
    device = data_loader[0][0].device
    dtype = data_loader[0][0].dtype
    data_loader1 = list()
    for img, labels in data_loader:
        img = img.reshape(img.shape[0], img.shape[1], dim1)
        tmp = tc.zeros((img.shape[0], img.shape[1]*size[0]*size[1]),
                       device=device, dtype=dtype)
        for n in range(img.shape[0]):
            _tmp = img[n, :, :].reshape(size0).cpu().numpy()
            _tmp = cv2.resize(_tmp.transpose(1, 2, 0), tuple(size))
            _tmp = _tmp.transpose(2, 0, 1).reshape(-1, )
            tmp[n, :] = tc.from_numpy(_tmp).to(device)
        data_loader1.append((tmp.reshape([img.shape[0]] + s1), labels))
    return data_loader1


def center_cut_img_size_loader(data_loader, size, dataset):
    # use this function before feature map
    if (dataset.lower() == 'mnist') or (dataset.lower() == 'fashion-mnist'):
        size0 = (28, 28)
    else:  # dataset.lower() == 'cifar10'
        size0 = (32, 32)
    if (size0[0] == size[0]) and (size0[1] == size[1]):
        return data_loader
    else:
        x0, x1 = round((size0[0] - size[0]) / 2), round((size0[1] - size[1]) / 2)
        data_loader1 = list()
        for img, labels in data_loader:
            if data_loader[0][0].ndimension() == 2:
                tmp = img.reshape((img.shape[0],) + size0)[:, x0:(x0+size[0]), x1:(x1+size[1])]
                tmp = tmp.reshape(-1, size[0]*size[1])
            elif data_loader[0][0].ndimension() == 3:
                tmp = img[:, x0:(x0+size[0]), x1:(x1+size[1])]
            else:  # data_loader[0][0].ndimension() == 4
                tmp = img[:, :, x0:(x0 + size[0]), x1:(x1 + size[1])]
            data_loader1.append((tmp, labels))
        return data_loader1


def map_x_vector_taylor(x, d, data_device, data_dtype):
    if x.ndimension() == 2:
        vecs = tc.ones((x.shape[0], d) + x.shape[1:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d):
            vecs[:, n, :] = x ** n
    elif x.ndimension() == 3:
        vecs = tc.ones((x.shape[0], d) + x.shape[1:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d):
            vecs[:, n, :, :] = x ** n
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d):
            vecs[:, :, n, :, :] = x ** n
    else:   # x.shape = num, channel, d, lx, ly
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d):
            vecs[:, :, :, n, :, :] = x ** n
    return vecs


def map_x_vector_h_reflect_taylor(x, d, data_device, data_dtype):
    x[x < 0.5] = x[x < 0.5] - 1
    vecs = map_x_vector_taylor(x, d, data_device, data_dtype)
    return vecs


def map_x_vector_shrink_taylor(x, d, data_device, data_dtype,
                               alpha=0.5, beta=0.25):
    x = x * alpha + beta
    vecs = map_x_vector_taylor(x, d, data_device, data_dtype)
    return vecs


def map_x_vector_taylor_from1(x, d, data_device, data_dtype):
    if x.ndimension() == 2:
        vecs = tc.ones((x.shape[0], d) + x.shape[1:],
                       dtype=data_dtype, device=data_device)
        for n in range(d):
            vecs[:, n, :] = x ** (n+1)
    elif x.ndimension() == 3:
        vecs = tc.ones((x.shape[0], d) + x.shape[1:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d):
            vecs[:, n, :, :] = x ** (n+1)
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d):
            vecs[:, :, n, :, :] = x ** (n+1)
    else:   # x.shape = num, channel, d, lx, ly
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d):
            vecs[:, :, :, n, :, :] = x ** (n+1)
    return vecs


def map_x_vector_shrink_taylor_from1(x, d, data_device, data_dtype,
                                     alpha=0.5, beta=0.25):
    x = x * alpha + beta
    vecs = map_x_vector_taylor_from1(x, d, data_device, data_dtype)
    return vecs


def map_x_vector_cos(x, d, data_device, data_dtype, theta_max):
    if x.ndimension() == 2:
        vecs = tc.ones((x.shape[0], d) + x.shape[1:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d):
            vecs[:, n, :] = tc.cos(x * n * np.pi / 2 * theta_max)
    elif x.ndimension() == 3:
        vecs = tc.ones((x.shape[0], d) + x.shape[1:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d):
            vecs[:, n, :, :] = tc.cos(x * n * np.pi / 2 * theta_max)
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d):
            vecs[:, :, n, :, :] = tc.cos(x * n * np.pi / 2 * theta_max)
    else:  # x.shape = num, channel, d, lx, ly
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d):
            vecs[:, :, :, n, :, :] = tc.cos(
                x * n * np.pi / 2 * theta_max)
    return vecs


def map_x_vector_cos_sin(x, d, data_device, data_dtype, theta_max=1):
    # x.shape = num_samples, length
    if x.ndimension() == 2:
        vecs = tc.ones((x.size()[0], d) + x.shape[1:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d+1):
            vecs[:, n - 1, :] = (math.sqrt(combination(d - 1, n - 1)) * (
                    tc.cos(x * theta_max * np.pi / 2) ** (d - n)) * (
                    tc.sin(x * theta_max * np.pi / 2) ** (n - 1)))
    elif x.ndimension() == 3:
        vecs = tc.ones((x.size()[0], d) + x.shape[1:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d+1):
            vecs[:, n - 1, :, :] = (math.sqrt(combination(d - 1, n - 1)) * (
                    tc.cos(x * theta_max * np.pi / 2) ** (d - n)) * (
                    tc.sin(x * theta_max * np.pi / 2) ** (n - 1)))
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d + 1):
            vecs[:, :, n - 1, :, :] = (math.sqrt(combination(d - 1, n - 1)) * (
                    tc.cos(x * theta_max * np.pi / 2) ** (d - n)) * (
                    tc.sin(x * theta_max * np.pi / 2) ** (n - 1)))
    else:
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        for n in range(1, d + 1):
            vecs[:, :, :, n - 1, :, :] = (math.sqrt(combination(d - 1, n - 1)) * (
                    tc.cos(x * theta_max * np.pi / 2) ** (d - n)) * (
                    tc.sin(x * theta_max * np.pi / 2) ** (n - 1)))
    return vecs


def map_x_vector_normalized_linear(x, d, data_device, data_dtype, theta_max=1):
    if d != 2:
        warning('To use the normalized linear map, d should be taken as 2')
        d = 2
    if (x.min() < 0) or (x.max() > 1):
        warning('You cannot use \'normalized_linear\' feature map'
                ' when x.min() < 0 or x.max() > 1')
    if x.ndimension() == 2:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :] = tc.sqrt(x * theta_max)
        vecs[:, 1, :] = tc.sqrt(1 - x * theta_max)
    elif x.ndimension() == 3:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :, :] = tc.sqrt(x * theta_max)
        vecs[:, 1, :, :] = tc.sqrt(1 - x * theta_max)
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, 0, :, :] = tc.sqrt(x * theta_max)
        vecs[:, :, 1, :, :] = tc.sqrt(1 - x * theta_max)
    else:
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, :, 0, :, :] = tc.sqrt(x * theta_max)
        vecs[:, :, :, 1, :, :] = tc.sqrt(1 - x * theta_max)
    return vecs


def map_x_RelSig(x, data_device, data_dtype, if_normalize):
    d = 2
    if x.ndimension() == 2:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :] = tc.sigmoid(x)
        vecs[:, 1, :] = nn.ReLU(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abc,ac->abc', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 3:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :, :] = tc.sigmoid(x)
        vecs[:, 1, :, :] = nn.ReLU(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abcd,acd->abcd', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, 0, :, :] = tc.sigmoid(x)
        vecs[:, :, 1, :, :] = nn.ReLU(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abcde,abde->abcde', [vecs, 1/(vecs.norm(dim=2)+1e-12)])
    else:
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, :, 0, :, :] = tc.sigmoid(x)
        vecs[:, :, :, 1, :, :] = nn.ReLU(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abcdef,abcef->abcdef', [vecs, 1/(vecs.norm(dim=3)+1e-12)])
    return vecs


def map_x_1MinusSig(x, data_device, data_dtype, if_normalize):
    d = 2
    if x.ndimension() == 2:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :] = tc.sigmoid(x)
        vecs[:, 1, :] = 1 - vecs[:, 0, :]
    elif x.ndimension() == 3:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :, :] = tc.sigmoid(x)
        vecs[:, 1, :, :] = 1 - vecs[:, 0, :, :]
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, 0, :, :] = tc.sigmoid(x)
        vecs[:, :, 1, :, :] = 1 - vecs[:, :, 0, :, :]
    else:
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, :, 0, :, :] = tc.sigmoid(x)
        vecs[:, :, :, 1, :, :] = 1 - vecs[:, :, :, 0, :, :]
    return vecs


def map_x_1RelSig(x, data_device, data_dtype, if_normalize):
    d = 3
    if x.ndimension() == 2:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :] = 1
        vecs[:, 1, :] = nn.ReLU(inplace=True)(x)
        vecs[:, 2, :] = tc.sigmoid(x)
        if if_normalize:
            vecs = tc.einsum('abc,ac->abc', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 3:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :, :] = 1
        vecs[:, 1, :, :] = nn.ReLU(inplace=True)(x)
        vecs[:, 2, :, :] = tc.sigmoid(x)
        if if_normalize:
            vecs = tc.einsum('abcd,acd->abcd', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, 0, :, :] = 1
        vecs[:, :, 1, :, :] = nn.ReLU(inplace=True)(x)
        vecs[:, :, 2, :, :] = tc.sigmoid(x)
        if if_normalize:
            vecs = tc.einsum('abcde,abde->abcde', [vecs, 1/(vecs.norm(dim=2)+1e-12)])
    else:
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, :, 0, :, :] = 1
        vecs[:, :, :, 1, :, :] = nn.ReLU(inplace=True)(x)
        vecs[:, :, :, 2, :, :] = tc.sigmoid(x)
        if if_normalize:
            vecs = tc.einsum('abcdef,abcef->abcdef', [vecs, 1/(vecs.norm(dim=3)+1e-12)])
    return vecs


def map_x_1RelSigCos(x, data_device, data_dtype, if_normalize):
    d = 4
    if x.ndimension() == 2:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :] = 1
        vecs[:, 1, :] = nn.ReLU(inplace=True)(x)
        vecs[:, 2, :] = tc.sigmoid(x)
        vecs[:, 3, :] = tc.cos(x * np.pi / 2)
        if if_normalize:
            vecs = tc.einsum('abc,ac->abc', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 3:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :, :] = 1
        vecs[:, 1, :, :] = nn.ReLU(inplace=True)(x)
        vecs[:, 2, :, :] = tc.sigmoid(x)
        vecs[:, 3, :, :] = tc.cos(x * np.pi / 2)
        if if_normalize:
            vecs = tc.einsum('abcd,acd->abcd', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, 0, :, :] = 1
        vecs[:, :, 1, :, :] = nn.ReLU(inplace=True)(x)
        vecs[:, :, 2, :, :] = tc.sigmoid(x)
        vecs[:, :, 3, :, :] = tc.cos(x * np.pi / 2)
        if if_normalize:
            vecs = tc.einsum('abcde,abde->abcde', [vecs, 1/(vecs.norm(dim=2)+1e-12)])
    else:
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, :, 0, :, :] = 1
        vecs[:, :, :, 1, :, :] = nn.ReLU(inplace=True)(x)
        vecs[:, :, :, 2, :, :] = tc.sigmoid(x)
        vecs[:, :, :, 3, :, :] = tc.cos(x * np.pi / 2)
        if if_normalize:
            vecs = tc.einsum('abcdef,abcef->abcdef', [vecs, 1/(vecs.norm(dim=3)+1e-12)])
    return vecs


def map_x_1RelCos(x, data_device, data_dtype, if_normalize):
    d = 3
    if x.ndimension() == 2:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :] = 1
        vecs[:, 1, :] = nn.ReLU(inplace=True)(x)
        vecs[:, 2, :] = tc.cos(x)
        if if_normalize:
            vecs = tc.einsum('abc,ac->abc', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 3:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :, :] = 1
        vecs[:, 1, :, :] = nn.ReLU(inplace=True)(x)
        vecs[:, 2, :, :] = tc.cos(x)
        if if_normalize:
            vecs = tc.einsum('abcd,acd->abcd', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, 0, :, :] = 1
        vecs[:, :, 1, :, :] = nn.ReLU(inplace=True)(x)
        vecs[:, :, 2, :, :] = tc.cos(x)
        if if_normalize:
            vecs = tc.einsum('abcde,abde->abcde', [vecs, 1/(vecs.norm(dim=2)+1e-12)])
    else:
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, :, 0, :, :] = 1
        vecs[:, :, :, 1, :, :] = nn.ReLU(inplace=True)(x)
        vecs[:, :, :, 2, :, :] = tc.cos(x)
        if if_normalize:
            vecs = tc.einsum('abcdef,abcef->abcdef', [vecs, 1/(vecs.norm(dim=3)+1e-12)])
    return vecs


def map_x_1Relu(x, data_device, data_dtype, if_normalize):
    d = 2
    if x.ndimension() == 2:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :] = 1
        vecs[:, 1, :] = nn.ReLU(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abc,ac->abc', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 3:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :, :] = 1
        vecs[:, 1, :, :] = nn.ReLU(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abcd,acd->abcd', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, 0, :, :] = 1
        vecs[:, :, 1, :, :] = nn.ReLU(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abcde,abde->abcde', [vecs, 1/(vecs.norm(dim=2)+1e-12)])
    else:
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, :, 0, :, :] = 1
        vecs[:, :, :, 1, :, :] = nn.ReLU(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abcdef,abcef->abcdef', [vecs, 1/(vecs.norm(dim=3)+1e-12)])
    return vecs


def map_x_1Relu6(x, data_device, data_dtype, if_normalize):
    d = 2
    if x.ndimension() == 2:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :] = 1
        vecs[:, 1, :] = nn.ReLU6(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abc,ac->abc', [vecs, 1 / (vecs.norm(dim=1) + 1e-12)])
    elif x.ndimension() == 3:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :, :] = 1
        vecs[:, 1, :, :] = nn.ReLU6(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abcd,acd->abcd', [vecs, 1 / (vecs.norm(dim=1) + 1e-12)])
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, 0, :, :] = 1
        vecs[:, :, 1, :, :] = nn.ReLU6(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abcde,abde->abcde', [vecs, 1 / (vecs.norm(dim=2) + 1e-12)])
    else:
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, :, 0, :, :] = 1
        vecs[:, :, :, 1, :, :] = nn.ReLU6(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abcdef,abcef->abcdef', [vecs, 1 / (vecs.norm(dim=3) + 1e-12)])
    return vecs


def map_x_1Sigm(x, data_device, data_dtype, if_normalize):
    d = 2
    if x.ndimension() == 2:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :] = 1
        vecs[:, 1, :] = tc.sigmoid(x)
        if if_normalize:
            vecs = tc.einsum('abc,ac->abc', [vecs, 1 / (vecs.norm(dim=1) + 1e-12)])
    elif x.ndimension() == 3:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :, :] = 1
        vecs[:, 1, :, :] = tc.sigmoid(x)
        if if_normalize:
            vecs = tc.einsum('abcd,acd->abcd', [vecs, 1 / (vecs.norm(dim=1) + 1e-12)])
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, 0, :, :] = 1
        vecs[:, :, 1, :, :] = tc.sigmoid(x)
        if if_normalize:
            vecs = tc.einsum('abcde,abde->abcde', [vecs, 1 / (vecs.norm(dim=2) + 1e-12)])
    else:
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, :, 0, :, :] = 1
        vecs[:, :, :, 1, :, :] = tc.sigmoid(x)
        if if_normalize:
            vecs = tc.einsum('abcdef,abcef->abcdef', [vecs, 1 / (vecs.norm(dim=3) + 1e-12)])
    return vecs


def map_x_TanSig(x, data_device, data_dtype, if_normalize):
    d = 2
    if x.ndimension() == 2:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :] = tc.sigmoid(x)
        vecs[:, 1, :] = tc.tanh(x)
        if if_normalize:
            vecs = tc.einsum('abc,ac->abc', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 3:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :, :] = tc.sigmoid(x)
        vecs[:, 1, :, :] = tc.tanh(x)
        if if_normalize:
            vecs = tc.einsum('abcd,acd->abcd', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, 0, :, :] = tc.sigmoid(x)
        vecs[:, :, 1, :, :] = tc.tanh(x)
        if if_normalize:
            vecs = tc.einsum('abcde,abde->abcde', [vecs, 1/(vecs.norm(dim=2)+1e-12)])
    else:
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, :, 0, :, :] = tc.sigmoid(x)
        vecs[:, :, :, 1, :, :] = tc.tanh(x)
        if if_normalize:
            vecs = tc.einsum('abcdef,abcef->abcdef', [vecs, 1/(vecs.norm(dim=3)+1e-12)])
    return vecs


def map_x_RelTanSig(x, data_device, data_dtype, if_normalize):
    d = 3
    if x.ndimension() == 2:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :] = tc.sigmoid(x)
        vecs[:, 1, :] = tc.tanh(x)
        vecs[:, 2, :] = nn.ReLU(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abc,ac->abc', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 3:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :, :] = tc.sigmoid(x)
        vecs[:, 1, :, :] = tc.tanh(x)
        vecs[:, 2, :, :] = nn.ReLU(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abcd,acd->abcd', [vecs, 1/(vecs.norm(dim=1)+1e-12)])
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, 0, :, :] = tc.sigmoid(x)
        vecs[:, :, 1, :, :] = tc.tanh(x)
        vecs[:, :, 2, :, :] = nn.ReLU(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abcde,abde->abcde', [vecs, 1/(vecs.norm(dim=2)+1e-12)])
    else:
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, :, 0, :, :] = tc.sigmoid(x)
        vecs[:, :, :, 1, :, :] = tc.tanh(x)
        vecs[:, :, :, 2, :, :] = nn.ReLU(inplace=True)(x)
        if if_normalize:
            vecs = tc.einsum('abcdef,abcef->abcdef', [
                vecs, 1/(vecs.norm(dim=3)+1e-12)])
    return vecs


def map_x_Vsigmoid(x, data_device, data_dtype):
    d = 2
    if x.ndimension() == 2:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :] = tc.sigmoid(x)
        vecs[:, 1, :] = 1 - vecs[:, 0, :]
    elif x.ndimension() == 3:
        vecs = tc.zeros((x.shape[0], d) + x.shape[1:],
                        device=data_device, dtype=data_dtype)
        vecs[:, 0, :, :] = tc.sigmoid(x)
        vecs[:, 1, :, :] = 1 - vecs[:, 0, :, :]
    elif x.ndimension() == 4:  # x.shape = num, channel, lx, ly
        vecs = tc.ones(x.shape[:2] + (d,) + x.shape[2:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, 0, :, :] = tc.sigmoid(x)
        vecs[:, :, 1, :, :] = 1 - vecs[:, :, 0, :, :]
    else:
        vecs = tc.ones(x.shape[:3] + (d,) + x.shape[3:],
                       dtype=data_dtype, device=data_device)
        vecs[:, :, :, 0, :, :] = tc.sigmoid(x)
        vecs[:, :, :, 1, :, :] = 1 - vecs[:, :, :, 0, :, :]
    return vecs


def fold_data_2d_order1(x, data_device, data_dtype):
    # x.shape = num, size_x, size_y
    d = 5
    num, size_x, size_y = x.shape
    size_x1, size_y1 = size_x + (size_x % 2), size_y + (size_y % 2)
    x1 = tc.zeros((num, size_x1, size_y1), device=x.device, dtype=x.dtype)
    x1[:, :size_x, :size_y] = x
    lx, ly = round(size_x1/2), round(size_y1/2)
    vecs = tc.ones((num, d, lx, ly),
                   device=data_device, dtype=data_dtype)
    for nx in range(lx):
        for ny in range(ly):
            vecs[:, 1:, nx, ny] = x1[:, 2*nx:(2+1)*nx, 2*ny:(2+1)*ny].reshape(num, 4)
    return vecs


def feature_map(x, d, f_map, data_device, data_dtype, theta_max=1):
    if f_map == 'taylor':
        x = map_x_vector_taylor(x, d, data_device, data_dtype)
    elif f_map == 'h_reflect_taylor':
        x = map_x_vector_h_reflect_taylor(x, d, data_device, data_dtype)
    elif f_map == 'shrink_taylor':
        x = map_x_vector_shrink_taylor(x, d, data_device, data_dtype)
    elif f_map == 'taylor_from1':
        x = map_x_vector_taylor_from1(x, d, data_device, data_dtype)
    elif f_map == 'shrink_taylor_from1':
        x = map_x_vector_shrink_taylor_from1(x, d, data_device, data_dtype)
    elif f_map == 'cos_sin':
        x = map_x_vector_cos_sin(x, d, data_device, data_dtype, theta_max)
    elif f_map == 'cos':
        x = map_x_vector_cos(x, d, data_device, data_dtype, theta_max)
    elif f_map == 'normalized_linear':
        x = map_x_vector_normalized_linear(x, d, data_device, data_dtype, theta_max)
    elif f_map.lower() == '1relu':
        x = map_x_1Relu(x, data_device, data_dtype, theta_max)
    elif f_map.lower() == '1relu6':
        x = map_x_1Relu6(x, data_device, data_dtype, theta_max)
    elif f_map.lower() == '1sig':
        x = map_x_1Sigm(x, data_device, data_dtype, theta_max)
    elif f_map.lower() == 'relsig':
        x = map_x_RelSig(x, data_device, data_dtype, theta_max)
    elif f_map.lower() == '1relsig':
        x = map_x_1RelSig(x, data_device, data_dtype, theta_max)
    elif f_map.lower() == '1relsigcos':
        x = map_x_1RelSigCos(x, data_device, data_dtype, theta_max)
    elif f_map.lower() == '1relcos':
        x = map_x_1RelCos(x, data_device, data_dtype, theta_max)
    elif f_map.lower() == '1minussig':
        x = map_x_1MinusSig(x, data_device, data_dtype, theta_max)
    elif f_map.lower() == 'tansig':
        x = map_x_TanSig(x, data_device, data_dtype, theta_max)
    elif f_map.lower() == 'reltansig':
        x = map_x_RelTanSig(x, data_device, data_dtype, theta_max)
    elif f_map.lower() == 'vsigmoid':
        x = map_x_Vsigmoid(x, data_device, data_dtype)
    elif f_map == 'fold_2d_order1':
        x = fold_data_2d_order1(x, data_device, data_dtype)
    return x


def feature_map_data_loader(data_loader, d, f_map, device, dtype, theta_max=1):
    data_loader1 = list()
    for imgs, labels in data_loader:
        vecs = feature_map(imgs, d, f_map, device, dtype, theta_max)
        data_loader1.append((vecs.to(device), labels.to(device)))
    return data_loader1


def number_samples_in_loader(data_loader):
    num = 0
    for _, labels in data_loader:
        num += labels.shape[0]
    return num


def recover_tensor_from_data_loader(data_loader):
    device, dtype = data_loader[0][0].device, data_loader[0][0].dtype
    samples = tc.zeros(0, device=device, dtype=dtype)
    labels = tc.zeros(0, device=device, dtype=data_loader[0][1].dtype)
    for imgs, lbs in data_loader:
        samples = tc.cat((samples, imgs), dim=0)
        labels = tc.cat((labels, lbs), dim=0)
    return samples, labels


def re_batch_data_loader(data_loader):
    batch_size = data_loader[0][0].shape[0]
    data_dim = data_loader[0][0].ndimension()
    samples, labels = recover_tensor_from_data_loader(data_loader)
    return batch_dataset(samples, labels, batch_size, data_dim,
                         labels2mat=False, normalize=1)


def labels_to_matrix(labels, channel):
    mat = tc.zeros((labels.numel(), channel), device=labels.device,
                   dtype=labels.dtype)
    for n in range(labels.numel()):
        mat[n, labels[n]] = 1
    return mat


def pad_x_copy_one_line(x):
    # x.shape: num, channel, sx, sy
    s = x.shape
    x1 = tc.zeros((s[0], s[1], s[2]+1, s[3]+1), device=x.device, dtype=x.dtype)
    x1[:, :, :s[2], :s[3]] = x
    x1[:, :, -1, :s[3]] = x[:, :, -1, :]
    x1[:, :, :s[2], -1] = x[:, :, :, -1]
    x1[:, :, -1, -1] = x[:, :, -1, -1]
    return x1


def multi_gpu_parallel_for_nn(model):
    # from torch import nn
    # if tc.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    # model.to('cuda:0')
    return model


def arrangement(n, m):
    return factorial(n) / factorial(n-m)


def combination(n, m):
    return int(arrangement(n, m) / factorial(m))


def ent_entropy(rho, tol=1e-18):
    s = tc.symeig(rho)[0]
    s[s < tol] = tol
    s = s.reshape(-1, 1)
    return -s.t().mm(tc.log(s))[0].item()


def trace_stack(level0=2):
    # print the line and file name where this function is used
    info = stack()
    ns = info.__len__()
    for ns in range(level0, ns):
        cprint('in ' + str(info[ns][1]) + ' at line ' + str(info[ns][2]), 'green')


def warning(string, if_trace_stack=False):
    cprint(string, 'magenta')
    if if_trace_stack:
        trace_stack(3)


class logger(object):

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt=None, clear_history=False):
        from logging import handlers
        if clear_history and os.path.isfile(filename):
            os.remove(filename)
        if fmt is None:
            fmt = '%(message)s'
        self.level_relations = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'crit': logging.CRITICAL
        }

        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(
            filename=filename, when=when, backupCount=backCount, encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


class logger_color(object):
    """
    Not fixed
    Examples:
        log = Log()
        log.debug("---测试开始----")
        log.info("操作步骤")
        log.warning("----测试结束----")
        log.error("----测试错误----")
    """
    def __init__(self, filename, fmt=None, clear_history=False):
        from logging import handlers
        if clear_history and os.path.isfile(filename):
            os.remove(filename)
        if fmt is None:
            fmt = '%(message)s'

        self.filename = filename
        self.level_relations = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'crit': logging.CRITICAL
        }
        self.log_colors_config = {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        }

        self.logger = logging.getLogger(filename)
        self.formatter = colorlog.ColoredFormatter(
            fmt, log_colors=self.log_colors_config)

    def __console(self, level, message):
        from logging.handlers import RotatingFileHandler

        fh = RotatingFileHandler(filename=self.filename, mode='a', maxBytes=1024 * 1024 * 5,
                                 backupCount=5, encoding='utf-8')  # 使用RotatingFileHandler类，滚动备份日志
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)

        # 创建一个StreamHandler,用于输出到控制台
        ch = colorlog.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

        if level == 'info':
            self.logger.info(message)
        elif level == 'debug':
            self.logger.debug(message)
        elif level == 'warning':
            self.logger.warning(message)
        elif level == 'error':
            self.logger.error(message)
        # 这两行代码是为了避免日志输出重复问题
        self.logger.removeHandler(ch)
        self.logger.removeHandler(fh)
        fh.close()  # 关闭打开的文件

    def debug(self, message):
        self.__console('debug', message)

    def info(self, message):
        self.__console('info', message)

    def warning(self, message):
        self.__console('warning', message)

    def error(self, message):
        self.__console('error', message)


def show_multiple_images_v1(imgs, lxy=None, titles=None, save_name=None, cmap=None, plot=False):
    if cmap is None:
        cmap = plt.cm.gray
    ni = imgs.shape[0]
    imgs = imgs.to('cpu').detach().numpy()
    if lxy is None:
        lx = int(np.sqrt(ni)) + 1
        ly = int(ni / lx) + 1
    else:
        lx, ly = tuple(lxy)
    plt.figure()
    for n in range(ni):
        plt.subplot(lx, ly, n + 1)
        tmp = imgs[n, :, :, :].squeeze()
        if tmp.ndim == 2:
            plt.imshow(tmp, cmap=cmap)
        else:
            plt.imshow(tmp)
        if titles is not None:
            plt.title(str(titles[n]))
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
    if type(save_name) is str:
        plt.savefig(save_name)
    if plot:
        plt.show()
    plt.close()

