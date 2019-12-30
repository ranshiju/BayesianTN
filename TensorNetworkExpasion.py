import torch as tc
from torch import nn
import numpy as np
import copy
import math
import sys
import os
import matplotlib.pyplot as plt
from collections.abc import Iterable
import BasicFun as bf
import PlotFun as pf
import CollectedTNN as ctnn


dev = bf.choose_device()
dtp = tc.float32


class MPSExpMachine1X(nn.Module):

    def __init__(self, d, length, chi, mps_init='randn', device=None, dtype=None):
        """
        use_env: whether save envs to calculate grad tensor by tensor; or the grads of all tensors
                  are calculated together by calculating loss once
        """
        from MatrixProductState import MPS_finite
        super(MPSExpMachine1X, self).__init__()
        self.length = length
        self.d = d
        self.chi = chi

        self.envl = list()
        self.envr = list()
        if device is None:
            self.device = dev
        else:
            self.device = device
        if dtype is None:
            dtype = dtp
        self.dtype = dtype

        self.para_mps = {
            'd': self.d,
            'length': self.length,
            'chi': self.chi,
            'device': self.device,
            'dtype': self.dtype,
            'decomp_way': 'qr',
            'mps_init': mps_init
        }
        self.tensors = eval('tc.' + self.para_mps['mps_init'] + '(' + str((
            self.length, self.chi, self.d, self.chi)) + ')')
        self.tensors = nn.Parameter(self.tensors.to(self.device))
        self.mps = MPS_finite(self.para_mps, if_ag=True)
        self.mps.input_tensors(self.tensors)
        self.norms = list(range(self.length))

    def map_x_vector_taylor(self, x):
        vecs = tc.ones((x.shape[0], self.d), dtype=self.dtype, device=self.device)
        for n in range(1, self.d):
            vecs[:, n] = x ** n
        return vecs

    def pre_normalize_tensors_with_x(self, vecs):
        with tc.no_grad():
            self.forward(vecs, normalize=True)
            for n in range(self.length):
                self.mps.tensors[n] /= (self.norms[n] + 1e-8)  # for stability

    def initialize_env(self, vecs):
        num_s = vecs.shape[0]
        with tc.no_grad():
            self.envl.append(tc.ones((num_s, self.mps.tensors[0].shape[0]),
                             device=self.device, dtype=self.dtype))
            for n in range(self.length):
                env = tc.einsum('na, ni,aib->nb', self.envl[-1], vecs, self.mps.tensors[n])
                norm = tc.norm(env)
                env /= (norm + 1e-8)
                self.mps.tensors[n] /= (norm + 1e-8)
                self.envl.append(env)
            self.envr = list(range(self.length))
            self.envr[-1] = tc.ones((num_s, self.mps.tensors[-1].shape[2]),
                                    device=self.device, dtype=self.dtype)
            for n in range(self.length-1, 0, -1):
                self.envr[n-1] = tc.einsum(
                    'nb,ni,aib->na', self.envr[n], vecs, self.mps.tensors[n])
                norm = tc.norm(self.envr[n-1])
                self.envr[n-1] /= (norm + 1e-8)
                self.mps.tensors[n] /= (norm + 1e-8)

    def update_envl(self, nt, vecs):
        with tc.no_grad():
            self.envl[nt + 1] = tc.einsum(
                'na,ni,aib->nb', self.envl[nt], vecs, self.mps.tensors[nt])

    def update_envr(self, nt, vecs):
        if nt > 0.5:
            with tc.no_grad():
                self.envr[nt - 1] = tc.einsum(
                    'nb,ni,aib->na', self.envr[nt], vecs,
                    self.mps.tensors[nt])

    def forward(self, vecs, normalize=False, way='simple', nt=None):
        y = None
        if way == 'simple':
            self.norms = list(range(self.length))
            y = tc.ones((vecs.shape[0], self.mps.tensors[0].shape[0]),
                        device=self.device, dtype=self.dtype)
            for n in range(self.length):
                y = tc.einsum('ni,na,aib->nb', vecs, y, self.mps.tensors[n])
                if normalize:
                    with tc.no_grad():
                        self.norms[n] = tc.norm(y)
                        y /= (self.norms[n] + 1e-10)
        elif way == 'env':
            y = tc.einsum(
                'na,aib,nb,ni->n', self.envl[nt],
                self.mps.tensors[nt], self.envr[nt], vecs)
        return y.flatten()

    @staticmethod
    def calculate_loss(y_true, y):
        return tc.norm(y - y_true) / len(y_true)

    def optimize_tensor_nt(self, nt, lr, normalize_grad=True):
        if normalize_grad:
            self.mps.tensors[nt].data -= \
                lr * self.mps.tensors[nt].grad / max(self.mps.tensors[nt].grad.norm(), 1e-12)
        else:
            self.mps.tensors[nt].data -= lr * self.mps.tensors[nt].grad
        self.mps.tensors[nt].grad.data.zero_()

    def optimization_simple(self, lr, normalize_grad=True):
        for n in range(self.length):
            self.optimize_tensor_nt(n, lr, normalize_grad)


class MPSExpMachine_multiX(nn.Module):

    def __init__(self, d, length, chi, mps_init='randn', use_env=False, device=None, dtype=None):
        """
        use_env: whether save envs to calculate grad tensor by tensor; or the grads of all tensors
                  are calculated together by calculating loss once
        """
        from MatrixProductState import MPS_finite
        super(MPSExpMachine_multiX, self).__init__()
        self.use_env = use_env

        self.d = d
        self.length = length  # num_features
        self.chi = chi
        self.envl = list()
        self.envr = list()

        if device is None:
            self.device = dev
        else:
            self.device = device
        if dtype is None:
            self.dtype = dtp

        self.para_mps = {
            'd': self.d,
            'length': self.length,
            'chi': self.chi,
            'device': self.device,
            'dtype': self.dtype,
            'mps_init': mps_init
        }
        self.mps = MPS_finite(self.para_mps, if_ag=True)
        if self.para_mps['mps_init'] == 'No.1':
            self.tensors = tc.zeros((self.length, self.chi, self.d, self.chi))
            self.tensors[:, :, 0, :] = 1
        elif self.para_mps['mps_init'] == 'No.2':
            self.tensors = tc.zeros((self.length, self.chi, self.d, self.chi))
            self.tensors[:, :, 1, :] = 1
        else:
            self.tensors = eval('tc.' + self.para_mps['mps_init'] + '(' + str((
                self.length, self.chi, self.d, self.chi)) + ')')
        self.tensors = nn.Parameter(self.tensors.type(self.para_mps['dtype']).to(self.device))
        self.mps.input_tensors(self.tensors)
        self.norms = list(range(self.length))

    @staticmethod
    def set_forward_para(para):
        para0 = {
            'feature_map': None,  # None means feature map has already used
            'get_grad_way': 'simple',
            'activate': None,
            'normalize_center': False}
        for x in para0:
            if x not in para:
                para[x] = para0[x]
        return para

    def initial_mps(self, init_way):
        self.mps.initialize_mps(init_way)

    def map_x_vector_taylor(self, x):
        vecs = tc.ones((x.shape[0], self.d, self.length), dtype=self.dtype, device=self.device)
        for n in range(1, self.d):
            vecs[:, n, :] = x ** n
        return vecs

    def map_x_vector_cos_sin(self, x, theta_max=1):
        # x.shape = num_samples, length
        vecs = tc.ones((x.size()[0], self.d, self.length),
                       dtype=self.dtype, device=self.device)
        for n in range(1, self.d+1):
            vecs[:, n - 1, :] = (math.sqrt(bf.combination(self.d - 1, n - 1)) * (
                    tc.cos(x * theta_max * np.pi / 2) ** (self.d - n)) * (
                    tc.sin(x * theta_max * np.pi / 2) ** (n - 1)))
        return vecs

    def map_x_vector_normalized_linear(self, x, theta_max=1):
        if self.d != 2:
            bf.warning('To use the normalized linear map, d should be taken as 2')
            self.d = 2
        if (x.min() < 0) or (x.max() > 1):
            bf.warning('You cannot use \'normalized_linear\' feature map'
                       ' when x.min() < 0 or x.max() > 1')
        vecs = tc.zeros((x.shape[0], self.d, x.shape[1]),
                        device=self.device, dtype=self.dtype)
        vecs[:, 0, :] = tc.sqrt(x * theta_max)
        vecs[:, 1, :] = tc.sqrt(1 - x * theta_max)
        return vecs

    def feature_map(self, x, f_map):
        if f_map == 'taylor':
            x = self.map_x_vector_taylor(x)
        elif f_map == 'cos_sin':
            x = self.map_x_vector_cos_sin(x)
        elif f_map == 'normalized_linear':
            x = self.map_x_vector_normalized_linear(x)
        return x

    def feature_map_data_loader(self, data_loader, f_map, device):
        data_loader1 = list()
        for imgs, labels in data_loader:
            vecs = self.feature_map(imgs, f_map)
            data_loader1.append((vecs.to(device), labels.to(device)))
        return data_loader1

    def initialize_env_to_center(self, nt, vecs, para_fw, normalize=False):
        env = dict()
        env['l'] = list()
        with tc.no_grad():
            env['l'] = list(range(self.length+1))
            env['l'][0] = tc.ones((vecs.shape[0], self.mps.tensors[0].shape[0]),
                                  device=self.device, dtype=self.dtype)
            for n in range(nt):
                tmp = tc.einsum('na,ni,aib->nb', env['l'][n], vecs[:, :, n],
                                self.mps.tensors[n])
                if normalize:
                    norm = tc.norm(tmp, dim=1)
                    tmp /= (norm.max() + 1e-12)
                    self.mps.tensors[n] /= (norm.max() + 1e-12)
                env['l'][n+1] = env
            env['r'] = list(range(self.length))
            env['r'][-1] = tc.ones((
                vecs.shape[0], self.mps.tensors[-1].shape[2]),
                device=self.device, dtype=self.dtype)
            for n in range(self.length-1, nt, -1):
                env['r'][n-1] = tc.einsum(
                    'nb,ni,aib->na', env['r'][n], vecs[:, :, n],
                    self.mps.tensors[n])
                env['r'][n-1] = self.activate(env['r'][n-1], para_fw['activate'])
                if normalize:
                    norm = tc.norm(env['r'][n-1], dim=1)
                    env['r'][n-1] /= (norm.max() + 1e-12)
                    self.mps.tensors[n] /= (norm.max() + 1e-12)
        return env

    def update_envl(self, nt, vecs, env):
        with tc.no_grad():
            env[nt + 1] = tc.einsum(
                'na,ni,aib->nb', self.envl[nt], vecs[:, :, nt],
                self.mps.tensors[nt])
        return env

    def update_envr(self, nt, vecs, env):
        if nt > 0.5:
            with tc.no_grad():
                env[nt - 1] = tc.einsum(
                    'nb,ni,aib->na', self.envr[nt], vecs[:, :, nt],
                    self.mps.tensors[nt])
        return env

    @staticmethod
    def activate(y, afun='tanh'):
        if afun is None:
            return y
        elif afun == 'tanh':
            return (tc.tanh(y) + 1) / 2
        elif afun == 'sigmoid':
            return 1 / (1 + tc.exp(-y))
        elif afun == 'normalize':
            return tc.einsum('na,n->na', y, 1/tc.norm(y, dim=1))

    def calculate_y_simple(self, nt, vecs, y, activate=None, normalize=False):
        y = tc.einsum('ni,na,aib->nb', vecs[:, :, nt], y,
                      self.mps.tensors[nt])
        if nt != (self.length-1):
            y = self.activate(y, activate)
        norm = None
        if normalize:
            with tc.no_grad():
                norm = tc.norm(y, dim=1).max()
                # norm = y.abs().mean() / 2
                y = y/norm
        return y, norm

    def calculate_y_by_env(self, nt, vecs, env, normalize_t=False):
        if normalize_t:
            y = tc.einsum(
                'na,aib,nb,ni->n', env['l'][nt],
                self.mps.tensors[nt] / (tc.norm(self.mps.tensors[nt]) + 1e-12),
                env['r'][nt], vecs[:, :, nt])
        else:
            y = tc.einsum(
                'na,aib,nb,ni->n', env['l'][nt],
                self.mps.tensors[nt], env['r'][nt], vecs[:, :, nt])
        return y

    @staticmethod
    def calculate_loss(y_true, y):
        num_s = y_true.numel()
        loss = tc.sum((y.squeeze() - y_true.type(y.dtype)) ** 2) / num_s
        return loss

    def forward(self, para, x, n=None, env=None):
        y = self.calculate_y(
            x, env=env, feature_map=para['feature_map'],
            get_grad_way=para['get_grad_way'],
            activate=para['activate'], nt=n,
            normalize_center=para['normalize_center'])
        return y.flatten()

    def calculate_y(self, x, env=None, feature_map='taylor', get_grad_way='simple',
                    activate=None, nt=None, normalize_center=False):
        x = self.feature_map(x, f_map=feature_map)
        if get_grad_way == 'simple':
            y = tc.ones((x.shape[0], self.mps.tensors[0].shape[0]),
                        device=self.device, dtype=self.dtype)
            for nt in range(self.length):
                y, _ = self.calculate_y_simple(
                    nt, x, y, activate=activate, normalize=False)
        elif get_grad_way == 'env':
            y = self.calculate_y_by_env(nt, x, env, normalize_t=normalize_center)
            # y = self.activate(y, activate)
        else:
            bf.warning('\'get_grad_way\' is not well defined')
            y = None
        return y

    def pre_normalize(self, x, para_forward):
        with tc.no_grad():
            y = tc.ones((x.shape[0], self.mps.tensors[0].shape[0]),
                        device=self.device, dtype=self.dtype)
            for nt in range(self.length):
                y, norm = self.calculate_y_simple(
                    nt, x, y, activate=para_forward['activate'], normalize=True)
                self.mps.tensors[nt] /= (norm + 1e-12)

    def optimize_tensors_simple(self, lr, normalize_grad=True):
        for n in range(self.length):
            self.optimize_tensor_nt(n, lr, normalize_grad)

    def optimize_tensor_nt(self, nt, lr, normalize_grad=True):
        if normalize_grad:
            self.mps.tensors[nt].data -= \
                lr * self.mps.tensors[nt].grad / (
                    tc.norm(self.mps.tensors[nt].grad) + 1e-12)
        else:
            self.mps.tensors[nt].data -= lr * self.mps.tensors[nt].grad
        self.mps.tensors[nt].grad.data.zero_()

    @staticmethod
    def num_correct(labels, y, num_classes):
        with tc.no_grad():
            y1 = y.abs().reshape(-1, 1)
            for n in range(1, num_classes):
                z = (y - n).abs().reshape(-1, 1)
                y1 = tc.cat((y1, z), dim=1)
            pred = y1.argmin(dim=1).type(tc.int64)
            nc = (labels == pred).sum().item()
        return nc, pred


class MPSExpMachine_multiX_multiC(nn.Module):
    """
    for multi-channels
    """
    def __init__(self, d, length, channel, chi, tensors=None, mps_init='randn',
                 use_env=False, device=None, dtype=None):
        from MatrixProductState import MPS_finite_multi_channel
        super(MPSExpMachine_multiX_multiC, self).__init__()
        self.d = d
        self.length = length  # num_features
        self.channel = channel  # normally number of classes
        self.chi = chi

        if device is None:
            self.data_device = dev
        else:
            self.data_device = device
        if dtype is None:
            self.data_dtype = dtp

        self.para_mps = {
            'd': self.d,
            'length': self.length,
            'channel': self.channel,
            'chi': self.chi,
            'device': self.data_device,
            'dtype': self.data_dtype,
            'mps_init': mps_init
        }
        self.mps = MPS_finite_multi_channel(self.para_mps, if_ag=True)
        if tensors is None:
            tensor_size = (self.channel, self.length, self.chi, self.d, self.chi)
            if self.para_mps['mps_init'] == 'No.1':
                self.tensors = tc.zeros(tensor_size)
                self.tensors[:, :, :, 0, :] = 1
            elif self.para_mps['mps_init'] == 'No.2':
                self.tensors = tc.zeros(tensor_size)
                self.tensors[:, :, :, 1, :] = 1
            else:
                self.tensors = eval('tc.' + self.para_mps['mps_init'] + '(' + str(tensor_size) + ')')
            self.tensors = nn.Parameter(self.tensors.type(self.data_dtype).to(self.data_device),
                                        requires_grad=True)
            self.mps.input_tensors(self.tensors)
        else:
            self.tensors = nn.Parameter(tensors.to(self.data_device),
                                        requires_grad=True)
            self.mps.input_tensors(tensors)
        self.norms = list(range(self.length))

        self.envl = list()
        self.envr = list()
        self.use_env = use_env

    def initial_mps(self, init_way):
        # !!! not fixed for multi-channels
        self.mps.initialize_mps(init_way)

    def map_x_vector_taylor(self, x):
        vecs = tc.ones((x.shape[0], self.d) + x.shape[1:],
                       dtype=self.data_dtype, device=self.data_device)
        if vecs.ndimension() == 3:
            for n in range(1, self.d):
                vecs[:, n, :] = x ** n
        else:
            for n in range(1, self.d):
                vecs[:, n, :, :] = x ** n
        return vecs

    def map_x_vector_cos(self, x):
        vecs = tc.ones((x.shape[0], self.d) + x.shape[1:],
                       dtype=self.data_dtype, device=self.data_device)
        if vecs.ndimension() == 3:
            for n in range(1, self.d):
                vecs[:, n, :] = tc.cos(x * n * np.pi / 2)
        else:
            for n in range(1, self.d):
                vecs[:, n, :, :] = tc.cos(x * n * np.pi / 2)
        return vecs

    def map_x_vector_cos_sin(self, x, theta_max=1):
        # x.shape = num_samples, length
        vecs = tc.ones((x.size()[0], self.d) + x.shape[1:],
                       dtype=self.data_dtype, device=self.data_device)
        if vecs.ndimension() == 3:
            for n in range(1, self.d+1):
                vecs[:, n - 1, :] = (math.sqrt(bf.combination(self.d - 1, n - 1)) * (
                        tc.cos(x * theta_max * np.pi / 2) ** (self.d - n)) * (
                        tc.sin(x * theta_max * np.pi / 2) ** (n - 1)))
        else:
            for n in range(1, self.d+1):
                vecs[:, n - 1, :, :] = (math.sqrt(bf.combination(self.d - 1, n - 1)) * (
                        tc.cos(x * theta_max * np.pi / 2) ** (self.d - n)) * (
                        tc.sin(x * theta_max * np.pi / 2) ** (n - 1)))
        return vecs

    def map_x_vector_normalized_linear(self, x, theta_max=1):
        if self.d != 2:
            bf.warning('To use the normalized linear map, d should be taken as 2')
            self.d = 2
        if (x.min() < 0) or (x.max() > 1):
            bf.warning('You cannot use \'normalized_linear\' feature map'
                       ' when x.min() < 0 or x.max() > 1')
        vecs = tc.zeros((x.shape[0], self.d) + x.shape[1:],
                        device=self.data_device, dtype=self.data_dtype)
        if vecs.ndimension() == 3:
            vecs[:, 0, :] = tc.sqrt(x * theta_max)
            vecs[:, 1, :] = tc.sqrt(1 - x * theta_max)
        else:
            vecs[:, 0, :, :] = tc.sqrt(x * theta_max)
            vecs[:, 1, :, :] = tc.sqrt(1 - x * theta_max)
        return vecs

    def feature_map(self, x, f_map):
        if f_map == 'taylor':
            x = self.map_x_vector_taylor(x)
        elif f_map == 'cos_sin':
            x = self.map_x_vector_cos_sin(x)
        elif f_map == 'cos':
            x = self.map_x_vector_cos(x)
        elif f_map == 'normalized_linear':
            x = self.map_x_vector_normalized_linear(x)
        return x

    def feature_map_data_loader(self, data_loader, f_map, device):
        data_loader1 = list()
        for imgs, labels in data_loader:
            vecs = self.feature_map(imgs, f_map)
            data_loader1.append((vecs.to(device), labels.to(device)))
        return data_loader1

    def initialize_env_to_center(self, nt, vecs, para_fw, normalize=False):
        # !!! not fixed for multi-channels
        env = dict()
        env['l'] = list()
        with tc.no_grad():
            env['l'] = list(range(self.length+1))
            env['l'][0] = tc.ones((vecs.shape[0], self.mps.tensors[0].shape[0]),
                                  device=self.data_device, dtype=self.data_dtype)
            for n in range(nt):
                tmp = tc.einsum('na,ni,aib->nb', env['l'][n], vecs[:, :, n],
                                self.mps.tensors[n])
                if normalize:
                    norm = tc.norm(tmp, dim=1)
                    tmp /= (norm.max() + 1e-12)
                    self.mps.tensors[n] /= (norm.max() + 1e-12)
                env['l'][n+1] = env
            env['r'] = list(range(self.length))
            env['r'][-1] = tc.ones((
                vecs.shape[0], self.mps.tensors[-1].shape[2]),
                device=self.data_device, dtype=self.data_dtype)
            for n in range(self.length-1, nt, -1):
                env['r'][n-1] = tc.einsum(
                    'nb,ni,aib->na', env['r'][n], vecs[:, :, n],
                    self.mps.tensors[n])
                env['r'][n-1] = self.activate(env['r'][n-1], para_fw['activate'])
                if normalize:
                    norm = tc.norm(env['r'][n-1], dim=1)
                    env['r'][n-1] /= (norm.max() + 1e-12)
                    self.mps.tensors[n] /= (norm.max() + 1e-12)
        return env

    def update_envl(self, nt, vecs, env):
        # !!! not fixed for multi-channels
        with tc.no_grad():
            env[nt + 1] = tc.einsum(
                'na,ni,aib->nb', self.envl[nt], vecs[:, :, nt],
                self.mps.tensors[nt])
        return env

    def update_envr(self, nt, vecs, env):
        # !!! not fixed for multi-channels
        if nt > 0.5:
            with tc.no_grad():
                env[nt - 1] = tc.einsum(
                    'nb,ni,aib->na', self.envr[nt], vecs[:, :, nt],
                    self.mps.tensors[nt])
        return env

    def calculate_y_simple(self, nt, vecs, y, activate=None, normalize=False):
        y = tc.einsum('ni,nca,caib->ncb', vecs[:, :, nt], y,
                      self.mps.tensors[nt])
        if nt != (self.length-1):
            y = self.activate(y, activate)
        norm = None
        if normalize:
            with tc.no_grad():
                norm = 1 / (y.norm(dim=2).max().item() + 1e-12)
                y *= norm
                # norm = 1 / (y.norm(dim=2).max(dim=0)[0] + 1e-12)
                # y = tc.einsum('ncb,c->ncb', y, norm)
        return y, norm

    def calculate_y_by_env(self, nt, vecs, env, normalize_t=False):
        # !!! not fixed for multi-channels
        if normalize_t:
            y = tc.einsum(
                'na,aib,nb,ni->n', env['l'][nt],
                self.mps.tensors[nt] / (tc.norm(self.mps.tensors[nt]) + 1e-12),
                env['r'][nt], vecs[:, :, nt])
        else:
            y = tc.einsum(
                'na,aib,nb,ni->n', env['l'][nt],
                self.mps.tensors[nt], env['r'][nt], vecs[:, :, nt])
        return y

    def forward(self, para, x, n=None, env=None):
        y = self.calculate_y(
            x, env=env, feature_map=para['feature_map'],
            get_grad_way=para['get_grad_way'],
            activate=para['activate'], nt=n,
            normalize_center=para['normalize_center'])
        return y

    def calculate_y(self, x, env=None, feature_map='taylor', get_grad_way='simple',
                    activate=None, nt=None, normalize_center=False):
        x = self.feature_map(x, f_map=feature_map)
        if get_grad_way == 'simple':
            y = tc.ones((x.shape[0], self.channel, self.mps.tensors[0].shape[1]),
                        device=self.data_device, dtype=self.data_dtype)
            for nt in range(self.length):
                y, _ = self.calculate_y_simple(
                    nt, x, y, activate=activate, normalize=False)
        elif get_grad_way == 'env':
            # !!! not fixed for multi-channels
            y = self.calculate_y_by_env(nt, x, env, normalize_t=normalize_center)
            # y = self.activate(y, activate)
        else:
            bf.warning('\'get_grad_way\' is not well defined')
            y = None
        return y

    def pre_normalize(self, x, para_forward):
        with tc.no_grad():
            y = tc.ones((x.shape[0], self.channel, self.mps.tensors[0].shape[1]),
                        device=self.data_device, dtype=self.data_dtype)
            for nt in range(self.length):
                y, norm = self.calculate_y_simple(
                    nt, x, y, activate=para_forward['activate'], normalize=True)
                self.mps.tensors[nt] *= norm
                # self.mps.tensors[nt].data = tc.einsum(
                #     'caib,c->caib', self.mps.tensors[nt].data, norm)

    def optimize_tensors_simple(self, lr, normalize_grad=True):
        for n in range(self.length):
            self.optimize_tensor_nt(n, lr, normalize_grad)

    def optimize_tensor_nt(self, nt, lr, normalize_grad=True):
        if normalize_grad:
            self.mps.tensors[nt].data -= \
                lr * self.mps.tensors[nt].grad / (
                    tc.norm(self.mps.tensors[nt].grad) + 1e-12)
        else:
            self.mps.tensors[nt].data -= lr * self.mps.tensors[nt].grad
        self.mps.tensors[nt].grad.data.zero_()

    def mps_zero_grad(self):
        self.mps.mps_zero_grad()

    @staticmethod
    def set_forward_para(para):
        para0 = {
            'feature_map': None,  # None means feature map has already used
            'get_grad_way': 'simple',
            'activate': None,
            'normalize_center': False}
        for x in para0:
            if x not in para:
                para[x] = para0[x]
        return para

    @staticmethod
    def activate(y, afun='tanh'):
        if afun is None:
            return y
        elif afun == 'tanh':
            return (tc.tanh(y) + 1) / 2
        elif afun == 'sigmoid':
            return 1 / (1 + tc.exp(-y))
        elif afun == 'normalize':
            return tc.einsum('na,n->na', y, 1 / tc.norm(y, dim=1))

    @staticmethod
    def calculate_loss(y_true, y):
        num_s = y_true.numel()
        loss = tc.sum((y.squeeze() - y_true.type(y.dtype)) ** 2) / num_s
        return loss

    @staticmethod
    def num_correct(labels, y):
        with tc.no_grad():
            pred = y.squeeze().abs().argmax(dim=1)
            l1 = labels.argmax(dim=1)
            nc = (l1 == pred).sum().item()
        return nc, pred


class MPS_BP(nn.Module):
    """
    For multi-channels;
    Only BP update
    """
    def __init__(self, para_tn, tensors=None, dtype=None):
        super(MPS_BP, self).__init__()
        self.d = para_tn['d']
        self.length = para_tn['length']  # num_features
        self.channel = para_tn['channel']  # normally number of classes
        self.chi = para_tn['chi']
        self.dims = list()

        if para_tn['device'] is None:
            self.data_device = dev
        else:
            self.data_device = para_tn['device']
        if dtype is None:
            self.data_dtype = dtp
        else:
            self.data_dtype = dtype

        self.get_virtual_dims()
        self.tensors = None
        if tensors is None:
            tensor_size = (self.channel, self.length, self.chi, self.d, self.chi)
            if para_tn['mps_init'] == 'No.1':
                self.tensors = tc.zeros(tensor_size)
                self.tensors[:, :, :, 0, :] = 1
                self.tensors[:, :, :, 1:, :] = 1e-10
                # self.tensors = initial_orthogonal_tensors_mps(
                #     self.tensors, self.channel, self.length, self.d, self.dims)
            elif para_tn['mps_init'] == 'No.2':
                self.tensors = tc.zeros(tensor_size)
                self.tensors[:, :, :, 1, :] = 1
            else:
                self.tensors = tc.zeros((
                    self.channel, self.length, self.chi, self.d, self.chi),
                    device=self.data_device, dtype=self.data_dtype)
                self.tensors = initial_orthogonal_tensors_mps(
                    self.tensors, self.channel, self.length, self.d, self.dims)
            self.tensors = nn.Parameter(self.tensors.type(self.data_dtype).to(
                self.data_device), requires_grad=True)
        else:
            self.tensors = tc.zeros((
                self.channel, self.length, self.chi, self.d, self.chi),
                device=self.data_device, dtype=self.data_dtype)
            s = tensors.shape
            self.tensors[:s[0], :s[1], :s[2], :s[3], :s[4]] = tensors
            self.tensors = nn.Parameter(self.tensors.to(self.data_device),
                                        requires_grad=True)
        self.norms = list(range(self.length))

    def get_virtual_dims(self):
        self.dims = list(np.ones((self.length + 1,), dtype=int))
        for n in range(0, self.length):
            if type(self.d) is int:
                chi1 = min([self.d ** (n + 1), self.chi, self.d ** (
                        self.length - n - 1)])
            else:
                chi1 = min([np.prod(self.d[:n + 1]), self.chi, np.prod(
                    self.d[:n + 1])])
            self.dims[n + 1] = chi1

    def calculate_y_simple(self, nt, vecs, y, activate=None, normalize=False):
        y = tc.einsum('nca,caib,ni->ncb', y,
                      self.tensors[:, nt, :self.dims[nt], :, :self.dims[nt+1]],
                      vecs[:, :, nt])
        if nt != (self.length-1):
            y = self.activate(y, activate)
        norm = None
        if normalize:
            with tc.no_grad():
                norm = 1 / (y.norm(dim=2).max().item() + 1e-12)
                y *= norm
        return y, norm

    def forward(self, x, activate=None):
        y = tc.ones((x.shape[0], self.channel, self.dims[0]),
                    device=self.data_device, dtype=self.data_dtype)
        for nt in range(self.length):
            y = self.calculate_y_simple(
                nt, x, y, activate=activate, normalize=False)[0]
        return y

    def pre_normalize(self, x, activate_f, way=0):
        with tc.no_grad():
            if way == 0:
                y = tc.ones((x.shape[0], self.channel, self.dims[0]),
                            device=self.data_device, dtype=self.data_dtype)
                for nt in range(self.length):
                    y, norm = self.calculate_y_simple(
                        nt, x, y, activate=activate_f, normalize=True)
                    self.tensors[:, nt, :self.dims[nt], :, :self.dims[nt+1]] = \
                        self.tensors[:, nt, :self.dims[nt], :, :self.dims[nt+1]] * norm
            elif way == 1:
                self.calculate_norm_tn(if_normalize=True)

    def pre_process_date_before_save(self):
        if self.tensors.grad is not None:
            self.tensors.grad.data.zero_()
            return self.tensors.data

    def calculate_norm_tensors(self, way):
        if way == 0:
            return self.tensors.norm()
        else:
            return self.tensors[:, :, :, 1:, :].norm()

    def calculate_norm_tn(self, if_normalize=False):
        y = tc.ones((self.channel, self.dims[0], self.dims[0]),
                    device=self.data_device, dtype=self.data_dtype)
        for nt in range(self.length):
            y = tc.einsum(
                'cad,caib,cdie->cbe',
                [y, self.tensors[:, nt, :self.dims[nt], :, :self.dims[nt + 1]],
                 self.tensors[:, nt, :self.dims[nt], :, :self.dims[nt + 1]]])
            with tc.no_grad():
                if if_normalize:
                    norm = (y.reshape(y.shape[0], -1).norm(dim=1) + 1e-12) * y.shape[1]
                    # norm = tc.abs(tc.einsum('cbe,be', y, tc.eye(
                    #     y.shape[1], device=self.data_device, dtype=self.data_dtype)))
                    # norm = norm * y.shape[1]
                    # norm = y.abs().reshape(y.shape[0], -1).max(dim=1)[0]
                    y = tc.einsum('cad,c->cad', y, 1/norm)
                    self.tensors[:, nt, :self.dims[nt], :, :self.dims[nt + 1]] =\
                        tc.einsum(
                            'caib,c->caib',
                            self.tensors[:, nt, :self.dims[nt], :, :self.dims[nt + 1]],
                            norm**(-0.5))
        return y.squeeze()

    def calculate_rho_onebody(self, tensors=None):
        with tc.no_grad():
            if tensors is None:
                y = [tc.ones((self.dims[0], self.dims[0]),
                             device=self.data_device, dtype=self.data_dtype)]
                for nt in range(self.length):
                    tmp = tc.einsum(
                        'ad,caib,cdie->be',
                        [y[-1], self.tensors[:, nt, :self.dims[nt], :, :self.dims[nt+1]],
                         self.tensors[:, nt, :self.dims[nt], :, :self.dims[nt+1]]]
                    )
                    tmp /= tc.norm(tmp)
                    y.append(tmp)
                yy = tc.ones((self.dims[self.length], self.dims[self.length]),
                             device=self.data_device, dtype=self.data_dtype)
                rho = [None] * self.length
                for nt in range(self.length-1, -1, -1):
                    rho[nt] = tc.einsum(
                        'ad,caib,cdje,be->ij',
                        [y[nt],
                         self.tensors.data[:, nt, :self.dims[nt], :, :self.dims[nt+1]],
                         self.tensors.data[:, nt, :self.dims[nt], :, :self.dims[nt+1]],
                         yy]
                    )
                    rho[nt] /= tc.trace(rho[nt])
                    if nt != 0:
                        yy = tc.einsum(
                            'be,caib,cdie->ad',
                            [yy, self.tensors.data[:, nt, :self.dims[nt], :, :self.dims[nt + 1]],
                             self.tensors.data[:, nt, :self.dims[nt], :, :self.dims[nt + 1]]])
                        yy /= tc.norm(yy)
            else:
                y = [tc.ones((self.dims[0], self.dims[0]),
                             device=self.data_device, dtype=self.data_dtype)]
                for nt in range(self.length):
                    tmp = tc.einsum(
                        'ad,caib,cdie->be',
                        [y[-1], tensors[:, nt, :self.dims[nt], :, :self.dims[nt + 1]],
                         tensors[:, nt, :self.dims[nt], :, :self.dims[nt + 1]]])
                    tmp /= tc.norm(tmp)
                    y.append(tmp)
                yy = tc.ones((self.dims[self.length], self.dims[self.length]),
                             device=self.data_device, dtype=self.data_dtype)
                rho = [None] * self.length
                for nt in range(self.length - 1, -1, -1):
                    rho[nt] = tc.einsum(
                        'ad,caib,cdje,be->ij',
                        [y[nt],
                         tensors[:, nt, :self.dims[nt], :, :self.dims[nt + 1]],
                         tensors[:, nt, :self.dims[nt], :, :self.dims[nt + 1]],
                         yy]
                    )
                    rho[nt] /= tc.trace(rho[nt])
                    if nt != 0:
                        yy = tc.einsum(
                            'be,caib,cdie->ad',
                            [yy, tensors[:, nt, :self.dims[nt], :, :self.dims[nt + 1]],
                             tensors[:, nt, :self.dims[nt], :, :self.dims[nt + 1]],])
                        yy /= tc.norm(yy)
            return rho

    def calculate_ent_onebody(self, rho=None):
        if rho is None:
            rho = self.calculate_rho_onebody()
        length = len(rho)
        ent = list()
        for n in range(length):
            ent.append(bf.ent_entropy(rho[n]))
        return ent

    def calculate_observable_onebody(self, op, rho):
        if rho is None:
            rho = self.calculate_rho_onebody()
        length = len(rho)


    @staticmethod
    def set_forward_para(para):
        para0 = {
            'feature_map': None,  # None means feature map has already used
            'activate': None,
            'normalize_center': False}
        for x in para0:
            if x not in para:
                para[x] = para0[x]
        return para

    @staticmethod
    def activate(y, afun='tanh'):
        if afun is None:
            return y
        elif afun == 'tanh':
            return (tc.tanh(y) + 1) / 2
        elif afun == 'sigmoid':
            return 1 / (1 + tc.exp(-y))
        elif afun == 'normalize':
            return tc.einsum('na,n->na', y, 1 / tc.norm(y, dim=1))

    @staticmethod
    def calculate_loss(y_true, y):
        num_s = y_true.numel()
        loss = tc.sum((y.squeeze() - y_true.type(y.dtype)) ** 2) / num_s
        return loss


class TNlayer_basic(nn.Module):

    def __init__(self, device=None, dtype=None, out_dims=5,
                 simple_channel=False, if_pre_proc_T=False, add_bias=True):
        super(TNlayer_basic, self).__init__()
        if device is None:
            self.data_device = dev
        else:
            self.data_device = device
        if dtype is None:
            self.data_dtype = dtp
        else:
            self.data_dtype = dtype
        self.tensors = tc.zeros(0, device=self.data_device)
        self.bias = tc.zeros(0, device=self.data_device)
        self.out_dims = out_dims
        self.simple_channel = simple_channel
        self.if_pre_proc_tensors = if_pre_proc_T
        self.add_bias = add_bias
        self.flag_tnn = None

    def initial_tensor_to_layer(self, shape_t, shape_b=None, ini_way='No.1'):
        if ini_way == 'No.1':
            tmp = tc.zeros(shape_t, device=self.data_device,
                           dtype=self.data_dtype) * 1e-8
            if len(shape_t) == 9:
                # tmp[:, :, :, 0, 0, 0, 0, :] = 1
                tmp[:, :, :, :, 1, 0, 0, 0, :] = tc.randn(
                    tmp[:, :, :, :, 1, 0, 0, 0, :].shape,
                    device=self.data_device, dtype=self.data_dtype)
                tmp[:, :, :, :, 0, 1, 0, 0, :] = tc.randn(
                    tmp[:, :, :, :, 1, 0, 0, 0, :].shape,
                    device=self.data_device, dtype=self.data_dtype)
                tmp[:, :, :, :, 0, 0, 1, 0, :] = tc.randn(
                    tmp[:, :, :, :, 1, 0, 0, 0, :].shape,
                    device=self.data_device, dtype=self.data_dtype)
                tmp[:, :, :, :, 0, 0, 0, 1, :] = tc.randn(
                    tmp[:, :, :, :, 1, 0, 0, 0, :].shape,
                    device=self.data_device, dtype=self.data_dtype)
            elif len(shape_t) == 7:
                # tmp[:, 0, 0, 0, 0, :] = 1
                tmp[:, :, 1, 0, 0, 0, :] = tc.randn(
                    tmp[:, :, 1, 0, 0, 0, :].shape,
                    device=self.data_device, dtype=self.data_dtype)
                tmp[:, :, 0, 1, 0, 0, :] = tc.randn(
                    tmp[:, :, 1, 0, 0, 0, :].shape,
                    device=self.data_device, dtype=self.data_dtype)
                tmp[:, :, 0, 0, 1, 0, :] = tc.randn(
                    tmp[:, :, 1, 0, 0, 0, :].shape,
                    device=self.data_device, dtype=self.data_dtype)
                tmp[:, :, 0, 0, 0, 1, :] = tc.randn(
                    tmp[:, :, 1, 0, 0, 0, :].shape,
                    device=self.data_device, dtype=self.data_dtype)
            else:
                tmp = tc.randn(shape_t, device=self.data_device,
                               dtype=self.data_dtype)
        elif ini_way == 'rand':
            tmp = tc.rand(shape_t, device=self.data_device,
                          dtype=self.data_dtype)
        else:  # ini_way == 'randn':
            tmp = tc.randn(shape_t, device=self.data_device,
                           dtype=self.data_dtype)
        self.tensors = nn.Parameter(tmp, requires_grad=True)
        if (shape_b is not None) and self.add_bias:
            self.bias = tc.randn(shape_b, device=self.data_device,
                                 dtype=self.data_dtype) * 1e-8
            self.bias = nn.Parameter(self.bias, requires_grad=True)

    def pre_process_tensors(self):
        if self.if_pre_proc_tensors == 'softmax':
            tensors = nn.Softmax(dim=self.tensors.ndimension()-1)(self.tensors)
            return tensors
        elif self.if_pre_proc_tensors == 'normalize':
            norm = self.tensors.norm(p=1, dim=self.tensors.ndimension()-1)
            exp = ''
            for n in range(self.tensors.ndimension()):
                exp += chr(97+n)
            exp += (',' + exp[:-1] + '->' + exp)
            tensors = tc.einsum(exp, [self.tensors.abs(), 1/(norm+1e-12)])
            return tensors
        elif self.if_pre_proc_tensors == 'square':
            return self.tensors ** 2
        else:
            return self.tensors


class MPS_tensor(TNlayer_basic):

    def __init__(self, para, c_in, c_out, length, din, dout, pos_out, device,
                 dtype=None, out_dims=5, simple_chl=False):
        super(MPS_tensor, self).__init__(
            device=device, dtype=dtype, out_dims=out_dims,
            simple_channel=simple_chl, add_bias=False)
        self.para = para
        self.c_in = c_in
        self.c_out = c_out
        self.length = length
        self.pos_out = pos_out
        self.din = din
        self.dout = dout
        self.normalization = para['normalize_tensors']

        if self.simple_channel:
            self.initial_tensor_to_layer(
                (c_in, length, dout, din, dout),
                (c_out, length, dout), ini_way=para['mps_init'])
        else:
            self.initial_tensor_to_layer(
                (c_in, c_out, length, dout, din, dout),
                (c_out, c_out, length, dout), ini_way=para['mps_init'])

    def forward(self, x):
        if x.ndimension() == 4:
            # assume from the output of NN
            num, channel0, sx, sy = x.shape
            x = x.view(num, channel0, 1, sx, sy)
            d = 1
        else:
            num, channel0, d, sx, sy = x.shape

        if sx * sy != self.length:
            bf.warning('Error: for MPS, it should be sx*sy = length.')
            sys.exit(-1)

        if self.simple_channel:
            yl = tc.einsum(
                'ncd,cdr->ncr',
                [x[:, :, :, 0, 0], self.tensors[:, 0, 0, :, :]])
            if self.add_bias:
                yl = yl + self.bias[:, 0, :].repeat(num, 1, 1)

            nt = 0
            for nx in range(sx):
                for ny in range(sy):
                    nt += 1
                    yl = tc.einsum(
                        'ncl,ncd,cldr->ncr',
                        [yl, x[:, :, :, nx, ny], self.tensors[:, nt, :, :, :]])
                    if self.add_bias:
                        yl = yl + self.bias[:, nt, :].repeat(num, 1, 1)
                    if nt == self.pos_out:
                        break

            yr = tc.einsum(
                'ncd,cdl->ncl',
                [x[:, :, :, -1, -1], self.tensors[:, 0, :, :, 0]])
            nt = self.length-1
            for nx in range(sx-1, -1, -1):
                for ny in range(sy-1, -1, -1):
                    nt -= 1
                    yr = tc.einsum(
                        'ncr,ncd,cldr->ncl',
                        [yr, x[:, :, :, nx, ny], self.tensors[:, nt, :, :, :]])
                    if nt == self.pos_out+1:
                        break

            y = tc.einsum('ncr,ncl->nc', [yl, yr])
        else:
            yl = tc.einsum(
                'ncd,chdr->nhr',
                [x[:, :, :, 0, 0], self.tensors[:, :, 0, 0, :, :]])

            nt = 0
            for nx in range(sx):
                for ny in range(sy):
                    nt += 1
                    yl = tc.einsum(
                        'ncl,ncd,chldr->nhr',
                        [yl, x[:, :, :, nx, ny], self.tensors[:, :, nt, :, :, :]])
                    if nt == self.pos_out:
                        break

            yr = tc.einsum(
                'ncd,chdl->nhl',
                [x[:, :, :, -1, -1], self.tensors[:, :, 0, :, :, 0]])
            nt = self.length - 1
            for nx in range(sx - 1, -1, -1):
                for ny in range(sy - 1, -1, -1):
                    nt -= 1
                    yr = tc.einsum(
                        'ncr,ncd,chldr->nhl',
                        [yr, x[:, :, :, nx, ny], self.tensors[:, :, nt, :, :, :]])
                    if nt == self.pos_out + 1:
                        break

            y = tc.einsum('ncr,ncl->nc', [yl, yr])
        return y


class TTN_Pool_2by2to1(TNlayer_basic):

    def __init__(self, c_in, c_out, nx, ny, din, dout, device,
                 dtype=None, ini_way='No.1', out_dims=5,
                 simple_chl=False, if_pre_proc_T=False, add_bias=True):
        super(TTN_Pool_2by2to1, self).__init__(
            device=device, dtype=dtype, out_dims=out_dims,
            simple_channel=simple_chl, if_pre_proc_T=if_pre_proc_T, add_bias=add_bias)
        self.c_in = c_in
        self.c_out = c_out
        self.nx = nx
        self.ny = ny
        self.din = din
        self.dout = dout
        if self.simple_channel:
            self.initial_tensor_to_layer(
                (c_in, nx, ny, din, din, din, din, dout),
                (c_out, nx, ny, dout), ini_way=ini_way)
        else:
            self.initial_tensor_to_layer(
                (c_in, c_out, nx, ny, din, din, din, din, dout),
                (c_out, nx, ny, dout), ini_way=ini_way)

    def forward(self, x):
        tensors = self.pre_process_tensors()
        if x.ndimension() == 4:
            # assume from the output of NN
            num, channel0, sx, sy = x.shape
            x = x.view(num, channel0, 1, sx, sy)
            d = 1
        else:
            num, channel0, d, sx, sy = x.shape
        sx1 = (sx // 2) + (sx % 2)
        sy1 = (sy // 2) + (sy % 2)
        flag_error = False
        if sx1 != self.nx:
            flag_error = True
            bf.warning('nx = %g in TTN_Pool_2by2to1 layer mismatch with data = %g'
                       % (self.nx, sx1))
        if sy1 != self.ny:
            flag_error = True
            bf.warning('nx = %g in TTN_Pool_2by2to1 layer mismatch with data = %g'
                       % (self.ny, sy1))
        if self.simple_channel and (self.c_in != self.c_out):
            flag_error = True
            bf.warning('Error: for simple channels, it requires c_in = c_out')
        if flag_error:
            sys.exit(1)
        x1 = tc.zeros((num, self.c_out, self.dout, sx1, sy1),
                      device=x.device, dtype=x.dtype)
        if self.simple_channel:
            for nx in range(sx1):
                for ny in range(sy1):
                    x1[:, :, :, nx, ny] = tc.einsum(
                        'nci,ncj,nck,ncl,cijklo->nco',
                        [x[:, :, :, nx * 2, ny * 2],
                         x[:, :, :, min(nx * 2 + 1, sx - 1), ny * 2],
                         x[:, :, :, nx * 2, min(ny * 2 + 1, sy - 1)],
                         x[:, :, :, min(nx * 2 + 1, sx - 1), min(ny * 2 + 1, sy - 1)],
                         tensors[:, :, nx, ny, :, :, :, :, :]])
                    if self.add_bias:
                        x1[:, :, :, nx, ny] = \
                            x1[:, :, :, nx, ny] + self.bias[:, nx, ny, :].repeat(
                                num, 1, 1)
        else:
            for nx in range(sx1):
                for ny in range(sy1):
                    x1[:, :, :, nx, ny] = tc.einsum(
                        'nci,ncj,nck,ncl,cpijklo->npo',
                        [x[:, :, :, nx*2, ny*2],
                         x[:, :, :, min(nx*2+1, sx-1), ny*2],
                         x[:, :, :, nx*2, min(ny*2+1, sy-1)],
                         x[:, :, :, min(nx*2+1, sx-1), min(ny*2+1, sy-1)],
                         tensors[:, :, nx, ny, :, :, :, :, :]])
                    if self.add_bias:
                        x1[:, :, :, nx, ny] = \
                            x1[:, :, :, nx, ny] + self.bias[:, nx, ny, :].repeat(
                                num, 1, 1)
        if self.out_dims != 5:
            s = x1.shape
            x1 = x1.view(s[0], s[1]*s[2], s[3], s[4])
        return x1


class TTN_PoolTI_2by2to1(TNlayer_basic):

    def __init__(self, c_in, c_out, din, dout, device, dtype=None,
                 ini_way='No.1', out_dims=5, simple_chl=False,
                 if_pre_proc_T=False, add_bias=True):
        super(TTN_PoolTI_2by2to1, self).__init__(
            device, dtype, out_dims=out_dims,
            simple_channel=simple_chl, if_pre_proc_T=if_pre_proc_T,
            add_bias=add_bias)
        self.c_in = c_in
        self.c_out = c_out
        self.din = din
        self.dout = dout
        if self.simple_channel:
            self.initial_tensor_to_layer(
                (c_in, din, din, din, din, dout), (c_out, dout),
                ini_way=ini_way)
        else:
            self.initial_tensor_to_layer(
                (c_in, c_out, din, din, din, din, dout), (c_out, dout),
                ini_way=ini_way)

    def forward(self, x):
        tensors = self.pre_process_tensors()
        if x.ndimension() == 4:
            # assume from the output of NN
            num, channel0, sx, sy = x.shape
            x = x.view(num, channel0, 1, sx, sy)
            d = 1
        else:
            num, channel0, d, sx, sy = x.shape
        sx1 = (sx // 2) + (sx % 2)
        sy1 = (sy // 2) + (sy % 2)
        if self.simple_channel and (self.c_in != self.c_out):
            bf.warning('Error: for simple channels, it requires c_in = c_out')

        x1 = tc.zeros((num, self.c_out, self.dout, sx1, sy1),
                      device=x.device, dtype=x.dtype)
        if self.simple_channel:
            for nx in range(sx1):
                for ny in range(sy1):
                    x1[:, :, :, nx, ny] = tc.einsum(
                        'nci,ncj,nck,ncl,cijklo->nco',
                        [x[:, :, :, nx*2, ny*2],
                         x[:, :, :, min(nx*2+1, sx-1), ny*2],
                         x[:, :, :, nx*2, min(ny*2+1, sy-1)],
                         x[:, :, :, min(nx*2+1, sx-1), min(ny*2+1, sy-1)],
                         tensors])
            if self.add_bias:
                x1 = x1 + self.bias.repeat(num, sx1, sy1, 1, 1).permute(
                    0, 3, 4, 1, 2)
        else:
            for nx in range(sx1):
                for ny in range(sy1):
                    x1[:, :, :, nx, ny] = tc.einsum(
                        'nci,ncj,nck,ncl,cpijklo->npo',
                        [x[:, :, :, nx*2, ny*2],
                         x[:, :, :, min(nx*2+1, sx-1), ny*2],
                         x[:, :, :, nx*2, min(ny*2+1, sy-1)],
                         x[:, :, :, min(nx*2+1, sx-1), min(ny*2+1, sy-1)],
                         tensors])
            if self.add_bias:
                x1 = x1 + self.bias.repeat(num, sx1, sy1, 1, 1).permute(
                    0, 3, 4, 1, 2)

        if self.out_dims != 5:
            s = x1.shape
            x1 = x1.view(s[0], s[1] * s[2], s[3], s[4])
        return x1


class TTN_Conv_2by2to1(TNlayer_basic):

    def __init__(self, c_in, c_out, nx, ny, din, dout, device, dtype=None,
                 ini_way='No.1', out_dims=5, simple_chl=False):
        super(TTN_Conv_2by2to1, self).__init__(
            device, dtype, out_dims=out_dims, simple_channel=simple_chl)
        self.c_in = c_in
        self.c_out = c_out
        self.nx = nx
        self.ny = ny
        self.din = din
        self.dout = dout
        if self.simple_channel:
            self.initial_tensor_to_layer(
                (c_in, nx, ny, din, din, din, din, dout),
                (c_out, nx, ny, dout),
                ini_way=ini_way)
        else:
            self.initial_tensor_to_layer(
                (c_in, c_out, nx, ny, din, din, din, din, dout),
                (c_out, nx, ny, dout),
                ini_way=ini_way)

    def forward(self, x):
        if x.ndimension() == 4:
            # assume from the output of NN
            num, channel0, sx, sy = x.shape
            x = x.view(num, channel0, 1, sx, sy)
            d = 1
        else:
            num, channel0, d, sx, sy = x.shape
        sx1 = sx - 1
        sy1 = sy - 1
        flag_error = False
        if sx1 != self.nx:
            flag_error = True
            bf.warning('nx = %g in TTN_Conv_2by2to1 layer mismatch with data = %g'
                       % (self.nx, sx1))
        if sy1 != self.ny:
            flag_error = True
            bf.warning('nx = %g in TTN_Conv_2by2to1 layer mismatch with data = %g'
                       % (self.ny, sy1))
        if self.simple_channel and (self.c_in != self.c_out):
            flag_error = True
            bf.warning('Error: for simple channels, it requires c_in = c_out')
        if flag_error:
            sys.exit(1)

        x1 = tc.zeros((num, self.c_out, self.dout, sx1, sy1),
                      device=x.device, dtype=x.dtype)
        if self.simple_channel:
            for nx in range(sx1):
                for ny in range(sy1):
                    x1[:, :, :, nx, ny] = tc.einsum(
                        'nci,ncj,nck,ncl,cijklo->nco',
                        [x[:, :, :, nx, ny],
                         x[:, :, :, nx+1, ny],
                         x[:, :, :, nx, ny+1],
                         x[:, :, :, nx+1, ny+1],
                         self.tensors[:, nx, ny, :, :, :, :, :]])
                    x1[:, :, :, nx, ny] = \
                        x1[:, :, :, nx, ny] + self.bias[:, nx, ny, :].repeat(
                            num, 1, 1)
        else:
            for nx in range(sx1):
                for ny in range(sy1):
                    x1[:, :, :, nx, ny] = tc.einsum(
                        'nci,ncj,nck,ncl,cpijklo->npo',
                        [x[:, :, :, nx, ny],
                         x[:, :, :, nx+1, ny],
                         x[:, :, :, nx, ny+1],
                         x[:, :, :, nx+1, ny+1],
                         self.tensors[:, nx, ny, :, :, :, :, :]])
                    x1[:, :, :, nx, ny] = \
                        x1[:, :, :, nx, ny] + self.bias[:, nx, ny, :].repeat(
                            num, 1, 1)

        if self.out_dims != 5:
            s = x1.shape
            x1 = x1.view(s[0], s[1] * s[2], s[3], s[4])
        return x1


class TTN_ConvTI_2by2to1(TNlayer_basic):

    def __init__(self, c_in, c_out, din, dout, device, dtype=None,
                 ini_way='No.1', out_dims=5, simple_chl=False,
                 if_pre_proc_T=False, add_bias=True):
        super(TTN_ConvTI_2by2to1, self).__init__(
            device, dtype, out_dims=out_dims, simple_channel=simple_chl,
            if_pre_proc_T=if_pre_proc_T, add_bias=add_bias)
        self.c_in = c_in
        self.c_out = c_out
        self.din = din
        self.dout = dout

        if self.simple_channel:
            self.initial_tensor_to_layer(
                (c_in, din, din, din, din, dout), (c_out, dout),
                ini_way=ini_way)
        else:
            self.initial_tensor_to_layer(
                (c_in, c_out, din, din, din, din, dout), (c_out, dout),
                ini_way=ini_way)

    def forward(self, x):
        tensors = self.pre_process_tensors()

        if x.ndimension() == 4:
            # assume from the output of NN
            num, channel0, sx, sy = x.shape
            x = x.view(num, channel0, 1, sx, sy)
            d = 1
        else:
            num, channel0, d, sx, sy = x.shape
        sx1 = sx - 1
        sy1 = sy - 1

        x1 = tc.zeros((num, self.c_out, self.dout, sx1, sy1),
                      device=x.device, dtype=x.dtype)
        if self.simple_channel:
            for nx in range(sx1):
                for ny in range(sy1):
                    x1[:, :, :, nx, ny] = tc.einsum(
                        'nci,ncj,nck,ncl,cijklo->nco',
                        [x[:, :, :, nx, ny],
                         x[:, :, :, nx+1, ny],
                         x[:, :, :, nx, ny+1],
                         x[:, :, :, nx+1, ny+1],
                         tensors])
            if self.add_bias:
                x1 = x1 + self.bias.repeat(num, sx1, sy1, 1, 1).permute(
                    0, 3, 4, 1, 2)
        else:
            for nx in range(sx1):
                for ny in range(sy1):
                    x1[:, :, :, nx, ny] = tc.einsum(
                        'nci,ncj,nck,ncl,cpijklo->npo',
                        [x[:, :, :, nx, ny],
                         x[:, :, :, nx+1, ny],
                         x[:, :, :, nx, ny+1],
                         x[:, :, :, nx+1, ny+1],
                         tensors])
            if self.add_bias:
                x1 = x1 + self.bias.repeat(num, sx1, sy1, 1, 1).permute(
                    0, 3, 4, 1, 2)

        if self.out_dims != 5:
            s = x1.shape
            x1 = x1.view(s[0], s[1] * s[2], s[3], s[4])
        return x1


class LinearChannel(TNlayer_basic):

    def __init__(self, channel_in, channel_out, device, dtype=None,
                 tensors=None, ini_way='No.1', out_dims=5, simple_chl=False):
        super(LinearChannel, self).__init__(
            device, dtype, out_dims=out_dims, simple_channel=simple_chl)
        self.channel_in = channel_in
        self.channel_out = channel_out

        self.initial_tensor_to_layer(
            tensors, (channel_in+1, channel_out),
            ini_way=ini_way)

    def forward(self, x):
        if x.ndimension() == 4:
            # assume from the output of NN
            num, channel, sx, sy = x.shape
            x = x.view(num, channel, 1, sx, sy)
            d = 1
        else:
            num, channel, d, sx, sy = x.shape
        x1 = x.permute(0, 2, 3, 4, 1).view(num*d*sx*sy, channel).mm(
            self.tensors[:-1, :]) + self.tensors[-1, :].repeat([num*d*sx*sy, 1])
        x1 = x1.view(num, d, sx, sy, self.channel_out).permute(0, 4, 1, 2, 3)
        if self.out_dims != 5:
            s = x1.shape
            x1 = x1.view(s[0], s[1]*s[2], s[3], s[4])
        return x1


class Vectorization(TNlayer_basic):

    def __init__(self, d, f_map, device, dtype=None, theta_m=1, out_dims=5):
        super(Vectorization, self).__init__(device, dtype, out_dims=out_dims)
        self.d = d
        self.f_map = f_map
        self.theta_m = theta_m

    def forward(self, x):
        x = bf.feature_map(x, self.d, self.f_map, self.data_device,
                           self.data_dtype, self.theta_m)
        if x.ndimension() == 6:
            s = x.shape
            x = x.view(s[0], s[1], s[2]*s[3], s[4], s[5])
        if self.out_dims != 5:
            s = x.shape
            x = x.view(s[0], s[1] * s[2], s[3], s[4])
        return x


# ============================================================
class TTN_basic(nn.Module):

    def __init__(self, num_layers):
        super(TTN_basic, self).__init__()
        self.num_layers = num_layers
        self.tensors = [None for _ in range(self.num_layers)]
        self.paras_group = list()
        self.layers_group = list()

    def input_tensors(self, tensors):
        if tensors is not None:
            self.load_state_dict(tensors)

    def input_tensors_TN_from_NN_Conv2D(
            self, nn_file, layers_correspond, device):
        """
        layers_correspond: {'NN_layer': 'TN_layer'}
        """
        state_dict_nn = bf.load(nn_file, 'tensors', device=device)
        for x in layers_correspond:
            layer_tn = layers_correspond[x]
            exec('self.' + layer_tn + '.tensors.data[:, :, 1, 0, 0, 0, 0] = '
                 'state_dict_nn[\'' + x + '.weight\'].data.permute(1, 0, 2, 3)[:, :, 0, 0]')
            exec('self.' + layer_tn + '.tensors.data[:, :, 0, 0, 1, 0, 0] = '
                 'state_dict_nn[\'' + x + '.weight\'].data.permute(1, 0, 2, 3)[:, :, 0, 1]')
            exec('self.' + layer_tn + '.tensors.data[:, :, 0, 1, 0, 0, 0] = '
                 'state_dict_nn[\'' + x + '.weight\'].data.permute(1, 0, 2, 3)[:, :, 1, 0]')
            exec('self.' + layer_tn + '.tensors.data[:, :, 0, 0, 0, 1, 0] = '
                 'state_dict_nn[\'' + x + '.weight\'].data.permute(1, 0, 2, 3)[:, :, 1, 1]')
            exec('self.' + layer_tn + '.bias.data[:, 0] = state_dict_nn[\'' +
                 x + '.bias\'].data')
            exec('self.' + layer_tn + '.tensors = nn.Parameter(self.' +
                 layer_tn + '.tensors, requires_grad=True)')
            exec('self.' + layer_tn + '.bias = nn.Parameter(self.' +
                 layer_tn + '.bias, requires_grad=True)')

    def input_tensors_NN_from_NN(
            self, nn_file, layers_correspond, device):
        state_dict_nn = bf.load(nn_file, 'tensors', device=device)
        for x in layers_correspond:
            layer_tn = layers_correspond[x]
            exec('self.state_dict()[\'' + layer_tn +
                 '.weight\'][:, :] = nn.Parameter(state_dict_nn[\'' + x + '.weight\'].data)')
            exec('self.state_dict()[\'' + layer_tn +
                 '.bias\'][:] = nn.Parameter(state_dict_nn[\'' + x + '.bias\'].data)')
            # exec('self.' + layer_tn + '.weight = nn.Parameter(self.' +
            #      layer_tn + '.weight, requires_grad=True)')
            # exec('self.' + layer_tn + '.bias = nn.Parameter(self.' +
            #      layer_tn + '.bias, requires_grad=True)')

            # exec('self.' + layer_tn + '.weight.data = '
            #                           'state_dict_nn[\'' + x + '.weight\'].data')
            # exec('self.' + layer_tn + '.bias.data = '
            #                           'state_dict_nn[\'' + x + '.bias\'].data')
            # exec('self.' + layer_tn + '.weight = nn.Parameter(self.' +
            #      layer_tn + '.weight, requires_grad=True)')
            # exec('self.' + layer_tn + '.bias = nn.Parameter(self.' +
            #      layer_tn + '.bias, requires_grad=True)')

    def forward(self, x):
        # x.shape = num, channel, d, lx, ly
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
        x = x.squeeze()
        return x

    def auto_set_paras_groups(self, lr, way='tn&nn'):
        if way == 'tn&nn':
            self.paras_group = [list(), list()]
            self.layers_group = [list(), list()]
            layer_name = ''
            for x in self.state_dict():
                x = x.split('.')[0]
                if x != layer_name:
                    if eval(judge_if_tn(x)):
                        self.paras_group[0].append(
                            {'params': eval(get_layer_parameters(x)), 'lr': lr}
                        )
                        self.layers_group[0].append(x)
                    else:
                        self.paras_group[1].append(
                            {'params': eval(get_layer_parameters(x)), 'lr': lr}
                        )
                        self.layers_group[1].append(x)
                    layer_name = x

    def manual_set_paras_groups(self, lr, groups):
        self.paras_group = [list() for _ in range(len(groups))]
        self.layers_group = [list() for _ in range(len(groups))]
        layer_name = ''
        for x in self.state_dict():
            x = x.split('.')[0]
            if x != layer_name:
                flag_assign = False
                for g in range(len(groups)):
                    if x in groups[g]:
                        self.paras_group[g].append(
                            {'params': eval(get_layer_parameters(x)), 'lr': lr}
                        )
                        self.layers_group[g].append(x)
                        flag_assign = True
                        break
                if not flag_assign:
                    bf.warning(x + ' not assigned to any optimizer. '
                                   'Auto-assign it to the last optimizer')
                    self.paras_group[-1].append(
                        {'params': eval(get_layer_parameters(x)), 'lr': lr}
                    )
                    self.layers_group[-1].append(x)
                layer_name = x

    def pre_process_data_dims(self, x):
        s = x.shape
        if hasattr(self.layer0, 'flag_tnn'):
            if x.ndimension() == 4:
                x = x.view(s[0], s[1], 1, s[2], s[3])
        else:
            if x.ndimension() == 5:
                x = x.view(s[0], s[1]*s[2], s[3], s[4])
        return x

    def pre_process_date_before_save(self):
        tensors = dict()
        for x in self.state_dict():
            tmp = copy.deepcopy(self.state_dict()[x].data)
            if tmp.grad is not None:
                tmp.grad.data.zero_()
            tensors[x] = tmp
        # for n in range(self.num_layers):
        #     if eval('hasattr(self.layer' + str(n) + ', \'tensors\')'):
        #         flag = eval('self.layer' + str(n) + '.tensors.grad is not None')
        #         if flag:
        #             exec('self.layer' + str(n) + '.tensors.grad.data.zero_()')
        #         tensors.append(eval('self.layer' + str(n) + '.tensors.data'))
        return tensors

    def calculate_norm_tensors(self, way=0, p=1):
        tensors = list(self.parameters())
        norm = 0
        if way == 0:
            for n in range(len(tensors)):
                norm = norm + tensors[n].norm(p=p)
            return norm
        else:
            for n in range(len(tensors)):
                norm1 = tensors[n].norm(p=p)
                norm = norm + (1-norm1).abs()
            return norm

    def pre_normalize(self, x, a_fun=None, way=None):
        pass
        # with tc.no_grad():
        #     for n in range(self.num_layers):
        #         x = eval('self.layer' + str(n) + '(x)')
        #         norm = x.abs().max()
        #         x = x / norm
        #         exec('self.layer' + str(n) + '.tensors.data /= norm')

    def normalize_all_tensors(self, way):
        with tc.no_grad():
            for x in self.state_dict():
                if way == 'softmax':
                    tensors = nn.Softmax(
                        dim=self.state_dict()[x].ndimension() - 1)(
                        self.state_dict()[x].data)
                elif way == 'norm1':
                    norm = self.state_dict()[x].norm(
                        p=1, dim=self.state_dict()[x].ndimension() - 1)
                    exp = ''
                    for n in range(self.state_dict()[x].ndimension()):
                        exp += chr(97 + n)
                    exp += (',' + exp[:-1] + '->' + exp)
                    tensors = tc.einsum(
                        exp, [self.state_dict()[x].abs(), 1 / (norm + 1e-12)])
                elif way == 'norm2':
                    norm = self.state_dict()[x].norm(
                        p=2, dim=self.state_dict()[x].ndimension() - 1)
                    exp = ''
                    for n in range(self.state_dict()[x].ndimension()):
                        exp += chr(97 + n)
                    exp += (',' + exp[:-1] + '->' + exp)
                    tensors = tc.einsum(
                        exp, [self.state_dict()[x].abs(), 1 / (norm + 1e-12)])
                exp = ':, ' * self.state_dict()[x].ndimension()
                exp = exp[:-1]
                exp = 'self.state_dict()[x].data[' + exp + '] = tensors.data'
                exec(exp)


class TTN1_BP(TTN_basic):

    def __init__(self, para_tn, tensors=None):
        super(TTN1_BP, self).__init__(num_layers=5)
        self.channel = para_tn['channel']
        self.layer0 = TTN_Pool_2by2to1(para_tn['channel'], 14, 14, para_tn['d'],
                                       para_tn['chi'], para_tn['device'])
        self.layer1 = TTN_Pool_2by2to1(para_tn['channel'], 7, 7, para_tn['chi'],
                                       para_tn['chi'], para_tn['device'])
        self.layer2 = TTN_Pool_2by2to1(para_tn['channel'], 4, 4, para_tn['chi'],
                                       para_tn['chi'], para_tn['device'])
        self.layer3 = TTN_Pool_2by2to1(para_tn['channel'], 2, 2, para_tn['chi'],
                                       para_tn['chi'], para_tn['device'])
        self.layer4 = TTN_Pool_2by2to1(para_tn['channel'], 1, 1, para_tn['chi'],
                                       1, para_tn['device'])
        self.input_tensors(tensors)


class TTN2_BP(TTN_basic):

    def __init__(self, para_tn, tensors=None):
        super(TTN2_BP, self).__init__(num_layers=6)
        self.channel = para_tn['channel']
        self.layer0 = TTN_Pool_2by2to1(para_tn['channel'], 14, 14, para_tn['d'],
                                       para_tn['chi'], para_tn['device'])
        self.layer1 = TTN_Pool_2by2to1(para_tn['channel'], 7, 7, para_tn['chi'],
                                       para_tn['chi'], para_tn['device'])
        self.layer2 = TTN_Conv_2by2to1(para_tn['channel'], 6, 6, para_tn['chi'],
                                       para_tn['chi'], para_tn['device'])
        self.layer3 = TTN_Pool_2by2to1(para_tn['channel'], 3, 3, para_tn['chi'],
                                       para_tn['chi'], para_tn['device'])
        self.layer4 = TTN_Conv_2by2to1(para_tn['channel'], 2, 2, para_tn['chi'],
                                       para_tn['chi'], para_tn['device'])
        self.layer5 = TTN_Conv_2by2to1(para_tn['channel'], 1, 1, para_tn['chi'],
                                       1, para_tn['device'])
        self.input_tensors(tensors)


class TTN3_1_BP(TTN_basic):

    def __init__(self, para_tn, tensors=None):
        super(TTN3_1_BP, self).__init__(num_layers=7)
        self.channel = para_tn['channel']
        self.layer0 = TTN_Pool_2by2to1(
            para_tn['channel'], 14, 14, para_tn['d'],
            para_tn['d'], para_tn['device'])
        self.layer1 = TTN_Conv_2by2to1(
            para_tn['channel'], 13, 13, para_tn['d'],
            para_tn['d'], para_tn['device'])
        self.layer2 = TTN_Pool_2by2to1(
            para_tn['channel'], 7, 7, para_tn['d'],
            para_tn['chi'], para_tn['device'])
        self.layer3 = TTN_Conv_2by2to1(
            para_tn['channel'], 6, 6, para_tn['chi'],
            para_tn['chi'], para_tn['device'])
        self.layer4 = TTN_Pool_2by2to1(
            para_tn['channel'], 3, 3, para_tn['chi'],
            para_tn['chi'], para_tn['device'])
        self.layer5 = TTN_Conv_2by2to1(
            para_tn['channel'], 2, 2, para_tn['chi'],
            para_tn['chi'], para_tn['device'])
        self.layer6 = TTN_Conv_2by2to1(
            para_tn['channel'], 1, 1, para_tn['chi'],
            1, para_tn['device'])
        self.input_tensors(tensors)


class TTN4_BP(TTN_basic):
    # vanishing problem unsolved
    def __init__(self, para_tn, tensors=None):
        super(TTN4_BP, self).__init__(num_layers=8)
        self.channel = para_tn['channel']
        self.layer0 = TTN_Conv_2by2to1(
            1, 27, 27, para_tn['d'],
            para_tn['d'], para_tn['device'])
        self.layer1 = TTN_Pool_2by2to1(
            para_tn['channel'], 14, 14, para_tn['d'],
            para_tn['d'], para_tn['device'])
        self.layer2 = TTN_Conv_2by2to1(
            para_tn['channel'], 13, 13, para_tn['d'],
            para_tn['d'], para_tn['device'])
        self.layer3 = TTN_Pool_2by2to1(
            para_tn['channel'], 7, 7, para_tn['d'],
            para_tn['chi'], para_tn['device'])
        self.layer4 = TTN_Conv_2by2to1(
            para_tn['channel'], 6, 6, para_tn['chi'],
            para_tn['chi'], para_tn['device'])
        self.layer5 = TTN_Pool_2by2to1(
            para_tn['channel'], 3, 3, para_tn['chi'],
            para_tn['chi'], para_tn['device'])
        self.layer6 = TTN_Conv_2by2to1(
            para_tn['channel'], 2, 2, para_tn['chi'],
            para_tn['chi'], para_tn['device'])
        self.layer7 = TTN_Conv_2by2to1(
            para_tn['channel'], 1, 1, para_tn['chi'],
            1, para_tn['device'])
        self.input_tensors(tensors)


class VLTTN2_BP(TTN_basic):
    """
    PPPP... (all non-TI); channel = 10
    High learnability; overfitting
                     train_acc   test_acc
    MNIST (d=2)       1           0.9613
    f-MNIST(d=2)      0.94815     0.8841
    f-MNIST(d=3)      1           0.8704
    """
    def __init__(self, para_tn, tensors=None):
        super(VLTTN2_BP, self).__init__(num_layers=9)
        self.channel = para_tn['channel']
        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer1 = TTN_Pool_2by2to1(
            para_tn['channel'], 14, 14, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer2 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer3 = TTN_Pool_2by2to1(
            para_tn['channel'], 7, 7, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer4 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer5 = TTN_Pool_2by2to1(
            para_tn['channel'], 4, 4, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer6 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer7 = TTN_Pool_2by2to1(
            para_tn['channel'], 2, 2, para_tn['d'],
            para_tn['d'], para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer8 = TTN_Conv_2by2to1(
            para_tn['channel'], 1, 1, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.activate_layers = [1, 3, 5]
        self.input_tensors(tensors)

    def forward(self, x):
        # x.shape = num, channel, d, lx, ly
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            if n in self.activate_layers:
                x = tc.sigmoid(x)
        x = x.squeeze()
        return x


class VLTTN4_2_BP(TTN_basic):
    """
    PPP... (all TI) + large channel
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)     0.9276      0.8965
    f-MNIST(d=3)     0.976933    0.9007
    """
    def __init__(self, para_tn, tensors=None):
        super(VLTTN4_2_BP, self).__init__(num_layers=11)
        self.channel = para_tn['channel']
        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer1 = TTN_PoolTI_2by2to1(
            10, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 14, 14
        self.layer2 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer3 = TTN_PoolTI_2by2to1(
            5, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 7, 7
        self.layer4 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer5 = TTN_PoolTI_2by2to1(
            5, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 4, 4
        self.layer6 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer7 = TTN_PoolTI_2by2to1(
            4, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer8 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer9 = TTN_PoolTI_2by2to1(
            5, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer10 = LinearChannel(
            5000, self.channel, para_tn['device'], ini_way=para_tn['mps_init'])
        self.input_tensors(tensors)
        self.activate_layers = [1, 3, 5, 7, 9]

    def forward(self, x):
        # x.shape = num, channel, d, lx, ly
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            if n in self.activate_layers:
                x = tc.sigmoid(x)
        x = x.squeeze()
        return x


class VLTTN4_4_BP(TTN_basic):
    """
    PPP... (all TI) + large channel
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VLTTN4_4_BP, self).__init__(num_layers=11)
        self.channel = para_tn['channel']
        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer1 = TTN_PoolTI_2by2to1(
            10, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 14, 14
        self.layer2 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer3 = TTN_PoolTI_2by2to1(
            10, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'], extend_channel=True)  # 7, 7
        self.layer4 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer5 = TTN_PoolTI_2by2to1(
            5, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 4, 4
        self.layer6 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer7 = TTN_PoolTI_2by2to1(
            4, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer8 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer9 = TTN_PoolTI_2by2to1(
            4, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer10 = LinearChannel(
            8000, self.channel, para_tn['device'], ini_way=para_tn['mps_init'])
        self.input_tensors(tensors)
        self.activate_layers = [1, 3, 5, 7, 9]

    def forward(self, x):
        # x.shape = num, channel, d, lx, ly
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            if n in self.activate_layers:
                x = tc.sigmoid(x)
        x = x.squeeze()
        return x


class VLTTN5_1_BP(TTN_basic):
    """
    PPP... (all TI) + large channel
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VLTTN5_1_BP, self).__init__(num_layers=11)
        self.channel = para_tn['channel']
        self.p = 1

        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer1 = TTN_PoolTI_2by2to1(
            10, para_tn['d'], para_tn['d'],
            para_tn['device'], ini_way=para_tn['mps_init'])  # 14, 14
        self.layer2 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer3 = TTN_PoolTI_2by2to1(
            5, para_tn['d'], para_tn['d'],
            para_tn['device'], ini_way=para_tn['mps_init'])  # 7, 7
        self.layer4 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer5 = TTN_PoolTI_2by2to1(
            4, para_tn['d'], para_tn['d'],
            para_tn['device'], ini_way=para_tn['mps_init'])  # 4, 4
        self.layer6 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer7 = TTN_PoolTI_2by2to1(
            5, para_tn['d'], para_tn['d'],
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer8 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer9 = TTN_PoolTI_2by2to1(
            4, para_tn['d'], para_tn['d'],
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer10 = LinearChannel(
            4000, self.channel, para_tn['device'], ini_way=para_tn['mps_init'])
        self.input_tensors(tensors)
        self.activate_layers = [1, 3, 5, 7, 9]

    def forward(self, x):
        # x.shape = num, channel, d, lx, ly
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            if n in self.activate_layers:
                x = Vdim_to_norm(x, self.p)
                x = tc.sigmoid(x)
        x = x.squeeze()
        return x


class VLTTN5_4_BP(TTN_basic):
    """
    PPP... (all TI) + large channel
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VLTTN5_4_BP, self).__init__(num_layers=15)
        self.channel = para_tn['channel']
        self.p = 1

        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer1 = TTN_ConvTI_2by2to1(
            2, para_tn['d'], para_tn['d'],
            para_tn['device'], ini_way=para_tn['mps_init'])  # 27, 27
        self.layer2 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer3 = TTN_ConvTI_2by2to1(
            2, para_tn['d'], para_tn['d'],
            para_tn['device'], ini_way=para_tn['mps_init'])  # 26, 26
        self.layer4 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer5 = TTN_PoolTI_2by2to1(
            5, para_tn['d'], para_tn['d'],
            para_tn['device'], ini_way=para_tn['mps_init'])  # 13, 13
        self.layer6 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer7 = TTN_PoolTI_2by2to1(
            5, para_tn['d'], para_tn['d'],
            para_tn['device'], ini_way=para_tn['mps_init'])  # 7, 7
        self.layer8 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer9 = TTN_PoolTI_2by2to1(
            2, para_tn['d'], para_tn['d'],
            para_tn['device'], ini_way=para_tn['mps_init'])  # 4, 4
        self.layer10 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer11 = TTN_PoolTI_2by2to1(
            2, para_tn['d'], para_tn['d'],
            para_tn['device'], ini_way=para_tn['mps_init'])  # 2, 2
        self.layer12 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'])
        self.layer13 = TTN_PoolTI_2by2to1(
            2, para_tn['d'], para_tn['d'],
            para_tn['device'], ini_way=para_tn['mps_init'])  # 1, 1
        self.layer14 = LinearChannel(
            400, self.channel, para_tn['device'], ini_way=para_tn['mps_init'])
        self.input_tensors(tensors)
        self.activate_layers = [1, 3, 5, 7, 9, 11, 13]

    def forward(self, x):
        # x.shape = num, channel, d, lx, ly
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            if n in self.activate_layers:
                x = Vdim_to_norm(x, self.p)
                x = tc.sigmoid(x)
        x = x.squeeze()
        return x


def train_data_1d(which, num):
    if which is 'rand_1d':
        noise = 0.1
        x = tc.arange(-1, 1, 2 / num, device=dev, dtype=dtp)
        y = tc.exp(x)
        y += tc.randn((num, ), device=dev, dtype=dtp) * noise
        return x, y


def data_nd(n):
    pass


def scatter_plot(x, y_true, y, marker='s'):
    if type(x) is tc.Tensor:
        if x.device != 'cpu':
            x = x.cpu()
        x = x.numpy()
    plt.figure()
    plt.plot(x, y.cpu().numpy(), marker=marker)
    plt.scatter(x, y_true)
    plt.show()


def assign_optimizers(which, paras_group, model_name):
    if len(paras_group) == 0:
        exp = ('tc.optim.' + which + '(' + model_name +
               '.parameters(), lr=para[\'lr\'])')
        return exp
    else:
        optimizers = [None] * len(paras_group)
        for n in range(len(paras_group)):
            optimizers[n] = eval('tc.optim.' + which)(paras_group[n])
        return optimizers


def control_require_grad(tn, nl):
    # only the nl-th group require grad
    for n in range(len(tn.layers_group)):
        if n == nl:
            for m in range(len(tn.layers_group[n])):
                for x in eval('tn.' + tn.layers_group[n][m] + '.parameters()'):
                    x.requires_grad = True
        else:
            for m in range(len(tn.layers_group[n])):
                for x in eval('tn.' + tn.layers_group[n][m] + '.parameters()'):
                    x.requires_grad = False
    return tn


def get_layer_parameters(layer):
    return 'self.' + layer + '.parameters()'


def judge_if_tn(layer):
    return 'hasattr(self.' + layer + ', \'flag_tnn\')'


def pre_process_dataset(data_loader, para, device):
    data_loader = bf.center_cut_img_size_loader(
        data_loader, para['cut_size'], para['dataset'])
    data_loader = bf.resize_loader_mnist(
        data_loader, para['cut_size'], para['img_size'])
    if para['binary_imgs']:
        data_loader = bf.binary_loader(data_loader)
    if para['TN'][:2] not in ['FC', 'VL']:
        # Cases where feature map will be done with layers (not here
        # )
        # FC: feature map by fully-connected layer
        # VL: feature map by vectorization layer
        data_loader = bf.feature_map_data_loader(
            data_loader, para['d'], para['feature_map'], device,
            tc.float32, para['feature_theta'])
    num_tot = bf.number_samples_in_loader(data_loader)
    return data_loader, num_tot


def add_linear_gaussian_noise(imgs, strength):
    if strength is not None:
        imgs += tc.randn(imgs.shape, device=imgs.device, dtype=imgs.dtype) * imgs * strength
    return imgs


def Vdim_to_norm(x, p=1):
    #  x.shape = num, channel, d, lx, ly
    return x.norm(dim=2, p=p, keepdim=True)


def normalize_layer_output(x):
    s = x.shape
    norm = x.reshape(s[0], -1).norm(dim=1)
    x = tc.einsum('abcde,a->abcde', [x, (norm+1e-12)**(-1)])
    return x


def log_fidelity_cos_sin(data_loader, imgs1, theta_m=1):
    if imgs1.ndimension() == 5:
        s = imgs1.shape
        imgs1 = imgs1.view(s[0], s[1]*s[2], s[3], s[4])
    for imgs, labels in data_loader:
        s = imgs.shape
        if imgs.ndimension() == 5:
            imgs = imgs.view(s[0], s[1]*s[2], s[3], s[4])
        # imgs0 = imgs[labels == which_c, :, :, :]
        # s = imgs.shape
        fid = 0
        for n in range(imgs1.shape[0]):
            tmp = (imgs - imgs1[n, :, :, :].repeat(s[0], 1, 1, 1))
            tmp = tc.cos(tmp * theta_m * np.pi / 2)
            tmp = tc.log(tmp+1e-10).sum() / s[0]
            fid = fid + tmp
        return fid / imgs1.shape[0]


def load_tensor_network(file, para):
    tensors, info, para1 = bf.load(file, ['tensors', 'info', 'para'], device=para['device'])
    para1['device'] = para['device']
    exec('from ' + para['which_TN_set'] + ' import ' + para['TN'] + '_BP')
    tn = eval(para['TN'] + '_BP(para1, tensors=tensors)')
    return tn, info, para1


def save_tensor_network(tn, para, info, path, file):
    # tensors = tn.pre_process_date_before_save()
    bf.save(path, file, [tn.state_dict(), para, info],
            ['tensors', 'para', 'info'])


def initial_orthogonal_tensors_mps(tensors, channel, length, d, dims):
    for c in range(channel):
        for n in range(length - 1):
            tmp = tensors[c, n, :dims[n], :, :dims[n + 1]]
            tmp = tc.qr(tmp.reshape(-1, tmp.shape[2]))[0]
            tensors[c, n, :dims[n], :, :dims[n + 1]] = \
                tmp.reshape(dims[n], d, dims[n + 1])
        tmp = tensors[c, length - 1, :dims[length - 1], :, :dims[length]]
        tensors[c, length - 1, :dims[length - 1], :, :dims[length]] = \
            tmp / tmp.norm()
    return tensors


def load_mps_and_test_mnist(file_name, batch_size=None):
    if (type(file_name) == list) or (type(file_name) == tuple):
        file_name = os.path.join(file_name[0], file_name[1])
    if not os.path.isfile(file_name):
        bf.warning('No such a file: \'' + file_name + '\'')
    tensors, para = bf.load(file_name, ['tensors', 'para'])
    para['device'] = dev

    labels2mat = (para['loss_func'] == 'MSELoss')
    if para['TN'] == 'MPS':
        data_dim = 2
    else:
        data_dim = 5
    train_loader, test_loader = bf.load_mnist_and_batch(
        para['dataset'], para['classes'], para['num_samples'], None, para['batch_size'],
        data_dim=data_dim, labels2mat=labels2mat, channel=len(para['classes']),
        project_name=para['project'], dev=para['device'])

    train_loader, train_num_tot = pre_process_dataset(
        train_loader, para, para['device'])
    test_loader, test_num_tot = pre_process_dataset(
        test_loader, para, para['device'])

    num_batch_train = len(train_loader)
    print('Num of training samples:\t' + str(train_num_tot))
    print('Num of testing samples:\t' + str(test_num_tot))
    print('Num of training batches:\t' + str(num_batch_train))
    print('Num of features:\t' + str(para['length']))
    print('Dataset finish processed...')

    print('=' * 20)
    print('TN: ' + para['TN'] + '_BP')
    print('=' * 20)
    tn = eval(para['TN'] + '_BP(para, tensors=tensors)')
    loss_func = eval('tc.nn.' + para['loss_func'] + '()')
    activate_func = None
    if para['activate_fun_final'] is not None:
        activate_func = eval('tc.' + para['activate_fun_final'])

    train_loss = 0
    nc = 0
    with tc.no_grad():
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(para['device']), labels.to(para['device'])
            y = tn(imgs).squeeze()
            if para['normalize_mps']:
                norm_t = tn.calculate_norm_mps()
                y = tc.einsum('nc,c->nc', y, 1 / (norm_t + 1e-12))
            if para['activate_fun_final'] is not None:
                y = activate_func(y)
            loss = loss_func(y, labels)
            train_loss += loss.data.item()
            if para['Lagrangian'] is not None:
                loss1 = tn.calculate_norm_tensors(para['Lagrangian_way'])
                coeff_norm = loss1.data.item()
                train_loss += loss1.data.item() * para['Lagrangian'] / imgs.shape[0] \
                              / para['chi'] / para['chi']
            else:
                with tc.no_grad():
                    coeff_norm = tn.calculate_norm_tensors(0).data.item()
            nc0, _ = num_correct(labels, y.data)
            nc += nc0
        nct = test_accuracy_mnist(
            tn, test_loader, para)
    print('Train loss = ' + str(train_loss))
    print('Train acc = ' + str(nc / train_num_tot))
    print('Test acc = ' + str(nct / test_num_tot))
    print('Norm of coefficients = ' + str(coeff_norm))
    return nc / test_num_tot, nct/test_num_tot


def num_correct(labels, y):
    with tc.no_grad():
        pred = y.squeeze().argmax(dim=1)
        if labels.ndimension() == 2:
            l1 = labels.argmax(dim=1)
            nc = (l1 == pred).sum().item()
        else:
            nc = (labels == pred).sum().item()
    return nc, pred


def test_accuracy_mnist(tn, test_loader, para):
    with tc.no_grad():
        activate_func = None
        if para['activate_fun_final'] is not None:
            activate_func = eval('tc.' + para['activate_fun_final'])
        nc = 0
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(para['device']), labels.to(para['device'])
            yt = tn(imgs)
            if para['normalize_mps']:
                norm_t = tn.calculate_norm_tn()
                yt = tc.einsum('nc,c->nc', yt, 1 / (norm_t + 1e-12))
            if para['activate_fun_final'] is not None:
                yt = activate_func(yt)
            nc0, _ = num_correct(labels, yt.data)
            nc += nc0
    return nc


def plot_ent_2d(ent, size):
    ent = np.array(ent)
    pf.surf(z=ent.reshape(size))


def example1_regression_tnem():
    para = dict()
    para['it_time'] = 500
    para['lr'] = 1e-2
    para['d'] = 2
    para['l'] = 6
    para['chi'] = 8
    para['num_x'] = 200
    para['mps_init'] = 'randn'
    para['normalize_grad'] = False

    mps = MPSExpMachine1X(para['d'], para['l'], para['chi'], para['mps_init'])
    loss_func = nn.MSELoss()
    optimizer = tc.optim.Adam(mps.parameters(), lr=para['lr'])
    x, y_true = train_data_1d('rand_1d', para['num_x'])

    vecs = mps.map_x_vector_taylor(x)
    mps.pre_normalize_tensors_with_x(vecs)
    for t in range(para['it_time']):
        y = mps(vecs)
        loss = loss_func(y_true, y.squeeze())
        if (t % 50) == 0:
            print('At the ' + str(t) + '-th step, loss = ' + str(loss.data.item()))
            pf.plot(x, y_true.cpu().numpy().reshape(x.shape),
                    y.data.cpu().numpy().reshape(x.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # mps.optimization_simple(para['lr'] * loss.data.item(), para['normalize_grad'])


def example2_regression_tnem_env():
    print('Note: Cannot use nn.optimizor for env-bradient updates')
    para = dict()
    para['central_ort'] = True
    para['it_time'] = 500
    para['lr'] = 1e-2
    para['d'] = 2
    para['l'] = 10
    para['chi'] = 8
    para['init_way'] = 'randn'
    para['num_x'] = 200
    para['normalize_mps'] = False  # False is better, without careful experiments
    para['normalize_grad'] = True

    mps = MPSExpMachine1X(para['d'], para['l'], para['chi'], para['init_way'])
    loss_func = nn.MSELoss()

    x, y_true = train_data_1d('rand_1d', para['num_x'])
    vecs = mps.map_x_vector_taylor(x)
    # mps.pre_normalize_tensors_with_x(vecs)
    if para['central_ort']:
        mps.mps.central_orthogonalization(0, True)
    mps.initialize_env(vecs)
    for t in range(para['it_time']):
        for n in range(para['l']):
            y = mps(vecs, False, 'env', n)
            loss = loss_func(y, y_true)
            loss.backward()
            mps.optimize_tensor_nt(n, para['lr']*loss.data.item(), para['normalize_grad'])
            if para['central_ort']:
                mps.mps.central_orthogonalization(min(n+1, para['l']-1), para['normalize_mps'])
            mps.update_envl(n, vecs)
        for n in range(para['l']-1, -1, -1):
            y = mps(vecs, False, 'env', n)
            loss = loss_func(y, y_true)
            loss.backward()
            mps.optimize_tensor_nt(n, para['lr'] * loss.data.item(), para['normalize_grad'])
            if para['central_ort']:
                mps.mps.central_orthogonalization(max(n-1, 0), para['normalize_mps'])
            mps.update_envr(n, vecs)
        if (t % 50) == 0:
            print('At the ' + str(t) + '-th step, loss = ' + str(loss.data.item()))
            pf.plot(x, y_true.cpu().numpy().reshape(x.shape),
                    y.data.to('cpu').numpy().reshape(x.shape))


def example1_regression_tnem_multi_x():
    para = dict()
    para['it_time'] = 500
    para['lr'] = 1e-2
    para['d'] = 2
    para['l'] = 10
    para['chi'] = 8
    para['num_x'] = 200
    para['mps_init'] = 'randn'
    para['normalize_grad'] = True
    para['feature_map'] = 'taylor'
    para['activate_fun'] = None

    para_fw = {
        'feature_map': None,  # means to input vecs directly
        'get_grad_way': 'simple',
        'activate': para['activate_fun'],
        'normalize_center': False}

    mps = MPSExpMachine_multiX(para['d'], para['l'], para['chi'], mps_init=para['mps_init'])
    optimizer = tc.optim.Adam(mps.parameters(), lr=para['lr'])
    loss_func = nn.MSELoss()
    para_fw = mps.set_forward_para(para_fw)

    x0, y_true = train_data_1d('rand_1d', para['num_x'])
    x = tc.zeros((x0.numel(), para['l']), device=dev, dtype=dtp)
    for n in range(para['l']):
        x[:, n] = x0
    vecs = mps.feature_map(x, para['feature_map'])

    mps.pre_normalize(vecs, para_fw)
    for t in range(para['it_time']):
        y = mps(x=vecs, para=para_fw)
        # y = func.tanh(y)  # activate
        loss = loss_func(y_true, y)
        if (t % 50) == 0:
            print('At the ' + str(t) + '-th step, loss = ' + str(loss.data.item()))
            pf.plot(x0, y_true.cpu().numpy().reshape(x0.shape),
                    y.data.to('cpu').numpy().reshape(x0.shape))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # mps.optimize_tensors_simple(para['lr'] * loss.data.item(), para['normalize_grad'])


def tnem_multi_x_binary_classifier_mnist():
    para = dict()
    para['dataset'] = 'mnist'
    para['classes'] = [0, 1]
    para['num_samples'] = [100] * para['classes'].__len__()
    para['batch_size'] = 2000
    para['img_size'] = [16, 16]

    para['it_time'] = 1000
    para['lr'] = 1e-4
    para['d'] = 2
    para['chi'] = 8

    para['mps_init'] = 'No.1'
    para['normalize_grad'] = True
    para['feature_map'] = 'taylor'
    para['activate_fun'] = None
    para['activate_fun_final'] = None

    para_fw = {
        'feature_map': None,  # means to input vecs directly
        'get_grad_way': 'simple',
        'activate': para['activate_fun'],
        'normalize_center': False}

    para['check_time'] = 5
    para['if_test'] = True
    para['length'] = para['img_size'][0] * para['img_size'][1]

    mps = MPSExpMachine_multiX(para['d'], para['length'], para['chi'], mps_init=para['mps_init'])
    optimizer = tc.optim.Adam(mps.parameters(), lr=para['lr'])
    # optimizer = tc.optim.SGD(mps.parameters(), lr=para['lr'], momentum=0.9)
    loss_func = nn.MSELoss()

    train_loader, test_loader = bf.load_mnist_and_batch(
        para['classes'], para['num_samples'], None, para['batch_size'],
        data_dim=2, project_name='TNML/')
    train_loader = bf.resize_loader_mnist(train_loader, para['img_size'])
    test_loader = bf.resize_loader_mnist(test_loader, para['img_size'])
    train_num_tot = bf.number_samples_in_loader(train_loader)
    test_num_tot = bf.number_samples_in_loader(test_loader)

    train_loader = mps.feature_map_data_loader(train_loader, para['feature_map'], 'cpu')
    test_loader = mps.feature_map_data_loader(test_loader, para['feature_map'], 'cpu')

    para_fw = mps.set_forward_para(para_fw)
    mps.pre_normalize(train_loader[0][0].to(dev), para_fw)
    num_batch = len(train_loader)
    for t in range(para['it_time']):
        train_loss = 0
        nc = 0
        if num_batch > 1:
            train_loader = bf.re_batch_data_loader(train_loader)
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(dev), labels.to(dev)
            y = mps(para_fw, imgs)
            if para['activate_fun_final'] is not None:
                y = para['activate_fun_final'](y)
            loss = loss_func(labels.type(dtp), y)
            train_loss += loss.data.item() * imgs.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # mps.optimize_tensors_simple(
            #     para['lr'] * loss.data.item(), para['normalize_grad'])
            if ((t + 1) % para['check_time']) == 0:
                nc0, _ = mps.num_correct(labels, y, para['classes'].__len__())
                nc += nc0
        if ((t+1) % para['check_time']) == 0:
            print('At the ' + str(t+1) + '-th step, train loss = ' + str(train_loss / train_num_tot))
            print(str(nc) + ' correct in ' + str(train_num_tot) + ' training samples')
            print('Train accuracy = ' + str(nc/train_num_tot))
            if para['if_test']:
                test_loss = 0
                nc = 0
                with tc.no_grad():
                    for img_t, label_t in test_loader:
                        img_t, label_t = img_t.to(dev), label_t.to(dev)
                        yt = mps(para_fw, img_t)
                        if para['activate_fun_final'] is not None:
                            yt = para['activate_fun_final'](yt)
                        loss = loss_func(label_t.type(dtp), yt)
                        test_loss += (loss.data.item() * img_t.shape[0])
                        nc0, pred0 = mps.num_correct(label_t, yt, para['classes'].__len__())
                        nc += nc0
                    print('Test loss = ' + str(test_loss / test_num_tot))
                    print('Test accuracy = ' + str(nc / test_num_tot))





