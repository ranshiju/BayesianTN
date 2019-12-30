import torch as tc
from torch import nn
import numpy as np
import copy
import BasicFun as bf
from torch.autograd import Variable

dev = bf.choose_device()
dtype = tc.float32


class MPS_finite(nn.Module):

    def __init__(self, para, if_ag):
        super(MPS_finite, self).__init__()
        self.if_ag = if_ag

        self.length = 0
        self.d = None
        self.chi = 0
        self.decomp_way = 'qr'
        self.device = None
        self.dtype = None

        self.input_parameters(para)

        self.tensors = None
        self.dims = None
        self.center = -1  # orthogonal center; -1 means no center
        self.lm = [tc.zeros(0, device=self.device, dtype=self.dtype)
                   for _ in range(0, self.length - 1)]
        self.ent = None

    def input_parameters(self, para):
        para_default = {
            'length': 10,
            'd': 2,
            'chi': 12,
            'device': None,
            'dtype': None,
            'decomp_way': 'qr'
        }
        for x in para_default:
            if x not in para:
                para[x] = para_default[x]
            exec('self.' + x + ' = ' + 'para[\'' + x + '\']')
        if self.device is None:
            self.device = dev
        if self.dtype is None:
            self.dtype = dtype

    def get_virtual_dims(self):
        self.dims = list(np.ones((self.length + 1,), dtype=int))
        for n in range(0, self.length):
            if type(self.d) is int:
                chi1 = min([self.d ** (n + 1), self.chi, self.d ** (self.length - n - 1)])
            else:
                chi1 = min([np.prod(self.d[:n + 1]), self.chi, np.prod(self.d[:n + 1])])
            self.dims[n + 1] = chi1

    def initial_mps(self, ini_way):
        self.tensors = list(range(self.length))
        if self.dims is None:
            self.get_virtual_dims()
        for n in range(self.length):
            chi0, chi1 = self.dims[n], self.dims[n + 1]
            self.tensors[n] = eval('tc.' + ini_way + '(' + str((chi0, self.d, chi1)) + ')')
            self.tensors[n] = self.tensors[n].type(self.dtype).to(self.device)
            self.tensors[n] = nn.Parameter(data=self.tensors[n])

    def input_tensors(self, tensors):
        # tensors.shape = length, chi, d, chi
        if self.dims is None:
            self.get_virtual_dims()
        self.tensors = list(range(self.length))
        for n in range(self.length):
            self.tensors[n] = tensors[n, :self.dims[n], :, :self.dims[n+1]]

    def orthogonalize_mps(self, l0, l1, normalize=False, is_trun=False, chi=-1):
        if is_trun:
            decomp_way = 'svd'  # if truncation is needed, it is mandatory to use 'svd'
        else:
            decomp_way = self.decomp_way
        with tc.no_grad():
            if l0 < l1:  # Orthogonalize MPS from left to right
                for n in range(l0, l1):
                    tensor, mat, self.dims[n + 1], lm = \
                        left2right_decompose_tensor(self.tensors[n].data, decomp_way)
                    if is_trun and (mat.shape[1] > chi):
                        self.tensors[n + 1].data = tc.einsum('aib,ax->xib', self.tensors[n + 1].data, mat[:, :chi])
                        tensor = tensor[:, :, :chi]
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n] = copy.copy(lm[:chi])
                        self.dims[n + 1] = chi
                    else:
                        self.tensors[n + 1].data = tc.einsum('aib,ax->xib', self.tensors[n + 1].data, mat)
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n] = copy.copy(lm)
                    if normalize:
                        self.tensors[n + 1].data /= max(tc.norm(self.tensors[n + 1].data).item(), 1e-12)
                    if self.if_ag:
                        self.tensors[n] = Variable(tensor, requires_grad=True)
            elif l0 > l1:  # Orthogonalize MPS from right to left
                for n in range(l0, l1, -1):
                    tensor, mat, self.dims[n], lm =\
                        right2left_decompose_tensor(self.tensors[n].data, decomp_way)
                    if is_trun and (mat.shape[1] > chi):
                        self.tensors[n - 1].data = tc.einsum('aib,bx->aix', self.tensors[n - 1].data, mat[:, :chi])
                        tensor = tensor[:chi, :, :]
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n - 1] = lm[:chi]
                        self.dims[n] = chi
                    else:
                        self.tensors[n - 1].data = tc.einsum('aib,bx->aix', self.tensors[n - 1].data, mat)
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n - 1] = copy.copy(lm)
                    if normalize:
                        self.tensors[n - 1].data /= max(tc.norm(self.tensors[n - 1].data).item(), 1e-12)
                    if self.if_ag:
                        self.tensors[n] = Variable(tensor, requires_grad=True)
        if self.if_ag:
            self.tensors[l1] = Variable(self.tensors[l1], requires_grad=True)

    def orthogonalize_mps_normal(self, l0, l1, normalize=False, is_trun=False, chi=-1):
        if is_trun:
            decomp_way = 'svd'  # if truncation is needed, it is mandatory to use 'svd'
        else:
            decomp_way = self.decomp_way
        with tc.no_grad():
            if l0 < l1:  # Orthogonalize MPS from left to right
                for n in range(l0, l1):
                    self.tensors[n], mat, self.dims[n + 1], lm = \
                        left2right_decompose_tensor(self.tensors[n], decomp_way)
                    if is_trun and (mat.shape[1] > chi):
                        self.tensors[n + 1] = tc.einsum('aib,ax->xib', self.tensors[n + 1], mat[:, :chi])
                        self.tensors[n] = self.tensors[n][:, :, :chi]
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n] = copy.copy(lm[:chi])
                        self.dims[n + 1] = chi
                    else:
                        self.tensors[n + 1] = tc.einsum('aib,ax->xib', self.tensors[n + 1], mat)
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n] = copy.copy(lm)
                    if normalize:
                        self.tensors[n + 1] /= max(tc.norm(self.tensors[n + 1]).item(), 1e-12)
                    if self.if_ag:
                        self.tensors[n] = Variable(self.tensors[n], requires_grad=True)
            elif l0 > l1:  # Orthogonalize MPS from right to left
                for n in range(l0, l1, -1):
                    self.tensors[n], mat, self.dims[n], lm =\
                        right2left_decompose_tensor(self.tensors[n], decomp_way)
                    if is_trun and (mat.shape[1] > chi):
                        self.tensors[n - 1] = tc.einsum('aib,bx->aix', self.tensors[n - 1], mat[:, :chi])
                        self.tensors[n] = self.tensors[n][:chi, :, :]
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n - 1] = lm[:chi]
                        self.dims[n] = chi
                    else:
                        self.tensors[n - 1] = tc.einsum('aib,bx->aix', self.tensors[n - 1], mat)
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n - 1] = copy.copy(lm)
                    if normalize:
                        self.tensors[n - 1] /= max(tc.norm(self.tensors[n - 1]).item(), 1e-12)
                    if self.if_ag:
                        self.tensors[n] = Variable(self.tensors[n], requires_grad=True)
        if self.if_ag:
            self.tensors[l1] = Variable(self.tensors[l1], requires_grad=True)

    def central_orthogonalization(self, nc, normalize=False):
        if self.center < 0:
            self.orthogonalize_mps(0, nc, normalize=normalize)
            self.orthogonalize_mps(self.length-1, nc, normalize=normalize)
        else:
            self.orthogonalize_mps(self.center, nc, normalize=normalize)
        self.center = nc

    def calculate_entanglement_spectrum(self, if_fast=True):
        # NOTE: this function will central orthogonalize the MPS
        _way = self.decomp_way
        _center = self.center
        self.decomp_way = 'svd'
        with tc.no_grad():
            if if_fast and _center > -0.5:
                p0 = self.length - 1
                p1 = 0
                for n in range(0, self.length - 1):
                    if self.lm[n].numel() == 0:
                        p0 = min(p0, n)
                        p1 = max(p1, n)
                self.central_orthogonalization(p0)
                self.central_orthogonalization(p1+1)
                self.central_orthogonalization(_center)
            else:
                self.central_orthogonalization(0)
                self.central_orthogonalization(self.length-1)
                if _center > 0:
                    self.central_orthogonalization(_center)
            self.decomp_way = _way

    def calculate_entanglement_entropy(self):
        self.ent = list(range(self.length-1))
        for i in range(0, self.length - 1):
            if self.lm[i].numel() == 0:
                self.ent[i] = -1
            else:
                self.ent[i] = self.entanglement_entropy(
                    self.lm[i].cpu().numpy())

    @staticmethod
    def entanglement_entropy(lm, tol=1e-20):
        with tc.no_grad():
            lm /= np.linalg.norm(lm)
            lm = lm[lm > tol]
            ent = -2 * (lm ** 2).T.dot(np.log(lm))
        return ent


class MPS_finite_multi_channel(nn.Module):

    def __init__(self, para, if_ag):
        super(MPS_finite_multi_channel, self).__init__()
        self.if_ag = if_ag

        self.length = 0
        self.d = None
        self.chi = 0
        self.channel = 0
        self.decomp_way = 'qr'
        self.device = None
        self.dtype = None

        self.input_parameters(para)

        self.tensors = None
        self.dims = None
        self.center = -1  # orthogonal center; -1 means no center
        self.lm = [tc.zeros(0, device=self.device, dtype=self.dtype)
                   for _ in range(0, self.length - 1)]
        self.ent = None

    def input_parameters(self, para):
        para_default = {
            'length': 10,
            'd': 2,
            'chi': 12,
            'channel': 1,
            'device': None,
            'dtype': None,
            'decomp_way': 'qr'
        }
        for x in para_default:
            if x not in para:
                para[x] = para_default[x]
            exec('self.' + x + ' = ' + 'para[\'' + x + '\']')
        if self.device is None:
            self.device = dev
        if self.dtype is None:
            self.dtype = dtype

    def get_virtual_dims(self):
        self.dims = list(np.ones((self.length + 1,), dtype=int))
        for n in range(0, self.length):
            if type(self.d) is int:
                chi1 = min([self.d ** (n + 1), self.chi, self.d ** (self.length - n - 1)])
            else:
                chi1 = min([np.prod(self.d[:n + 1]), self.chi, np.prod(self.d[:n + 1])])
            self.dims[n + 1] = chi1

    def initial_mps(self, ini_way):
        self.tensors = list(range(self.length))
        if self.dims is None:
            self.get_virtual_dims()
        for n in range(self.length):
            chi0, chi1 = self.dims[n], self.dims[n + 1]
            self.tensors[n] = eval('tc.' + ini_way + '(' + str((chi0, self.d, chi1)) + ')')
            self.tensors[n] = self.tensors[n].type(self.dtype).to(self.device)
            self.tensors[n] = nn.Parameter(data=self.tensors[n])

    def input_tensors(self, tensors):
        # tensors.shape = channel, length, chi, d, chi
        if self.dims is None:
            self.get_virtual_dims()
        self.tensors = list(range(self.length))
        for n in range(self.length):
            self.tensors[n] = tensors[:, n, :self.dims[n], :, :self.dims[n+1]]

    def orthogonalize_mps(self, l0, l1, normalize=False, is_trun=False, chi=-1):
        # !!! not fixed for multi-channels
        if is_trun:
            decomp_way = 'svd'  # if truncation is needed, it is mandatory to use 'svd'
        else:
            decomp_way = self.decomp_way
        with tc.no_grad():
            if l0 < l1:  # Orthogonalize MPS from left to right
                for n in range(l0, l1):
                    tensor, mat, self.dims[n + 1], lm = \
                        left2right_decompose_tensor(self.tensors[n].data, decomp_way)
                    if is_trun and (mat.shape[1] > chi):
                        self.tensors[n + 1].data = tc.einsum('aib,ax->xib', self.tensors[n + 1].data, mat[:, :chi])
                        tensor = tensor[:, :, :chi]
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n] = copy.copy(lm[:chi])
                        self.dims[n + 1] = chi
                    else:
                        self.tensors[n + 1].data = tc.einsum('aib,ax->xib', self.tensors[n + 1].data, mat)
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n] = copy.copy(lm)
                    if normalize:
                        self.tensors[n + 1].data /= max(tc.norm(self.tensors[n + 1].data).item(), 1e-12)
                    if self.if_ag:
                        self.tensors[n] = Variable(tensor, requires_grad=True)
            elif l0 > l1:  # Orthogonalize MPS from right to left
                for n in range(l0, l1, -1):
                    tensor, mat, self.dims[n], lm =\
                        right2left_decompose_tensor(self.tensors[n].data, decomp_way)
                    if is_trun and (mat.shape[1] > chi):
                        self.tensors[n - 1].data = tc.einsum('aib,bx->aix', self.tensors[n - 1].data, mat[:, :chi])
                        tensor = tensor[:chi, :, :]
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n - 1] = lm[:chi]
                        self.dims[n] = chi
                    else:
                        self.tensors[n - 1].data = tc.einsum('aib,bx->aix', self.tensors[n - 1].data, mat)
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n - 1] = copy.copy(lm)
                    if normalize:
                        self.tensors[n - 1].data /= max(tc.norm(self.tensors[n - 1].data).item(), 1e-12)
                    if self.if_ag:
                        self.tensors[n] = Variable(tensor, requires_grad=True)
        if self.if_ag:
            self.tensors[l1] = Variable(self.tensors[l1], requires_grad=True)

    def orthogonalize_mps_normal(self, l0, l1, normalize=False, is_trun=False, chi=-1):
        # !!! not fixed for multi-channels
        if is_trun:
            decomp_way = 'svd'  # if truncation is needed, it is mandatory to use 'svd'
        else:
            decomp_way = self.decomp_way
        with tc.no_grad():
            if l0 < l1:  # Orthogonalize MPS from left to right
                for n in range(l0, l1):
                    self.tensors[n], mat, self.dims[n + 1], lm = \
                        left2right_decompose_tensor(self.tensors[n], decomp_way)
                    if is_trun and (mat.shape[1] > chi):
                        self.tensors[n + 1] = tc.einsum('aib,ax->xib', self.tensors[n + 1], mat[:, :chi])
                        self.tensors[n] = self.tensors[n][:, :, :chi]
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n] = copy.copy(lm[:chi])
                        self.dims[n + 1] = chi
                    else:
                        self.tensors[n + 1] = tc.einsum('aib,ax->xib', self.tensors[n + 1], mat)
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n] = copy.copy(lm)
                    if normalize:
                        self.tensors[n + 1] /= max(tc.norm(self.tensors[n + 1]).item(), 1e-12)
                    if self.if_ag:
                        self.tensors[n] = Variable(self.tensors[n], requires_grad=True)
            elif l0 > l1:  # Orthogonalize MPS from right to left
                for n in range(l0, l1, -1):
                    self.tensors[n], mat, self.dims[n], lm =\
                        right2left_decompose_tensor(self.tensors[n], decomp_way)
                    if is_trun and (mat.shape[1] > chi):
                        self.tensors[n - 1] = tc.einsum('aib,bx->aix', self.tensors[n - 1], mat[:, :chi])
                        self.tensors[n] = self.tensors[n][:chi, :, :]
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n - 1] = lm[:chi]
                        self.dims[n] = chi
                    else:
                        self.tensors[n - 1] = tc.einsum('aib,bx->aix', self.tensors[n - 1], mat)
                        if lm.numel() > 0 and self.center > -1:
                            self.lm[n - 1] = copy.copy(lm)
                    if normalize:
                        self.tensors[n - 1] /= max(tc.norm(self.tensors[n - 1]).item(), 1e-12)
                    if self.if_ag:
                        self.tensors[n] = Variable(self.tensors[n], requires_grad=True)
        if self.if_ag:
            self.tensors[l1] = Variable(self.tensors[l1], requires_grad=True)

    def central_orthogonalization(self, nc, normalize=False):
        # !!! not fixed for multi-channels
        if self.center < 0:
            self.orthogonalize_mps(0, nc, normalize=normalize)
            self.orthogonalize_mps(self.length-1, nc, normalize=normalize)
        else:
            self.orthogonalize_mps(self.center, nc, normalize=normalize)
        self.center = nc

    def calculate_entanglement_spectrum(self, if_fast=True):
        # !!! not fixed for multi-channels
        # NOTE: this function will central orthogonalize the MPS
        _way = self.decomp_way
        _center = self.center
        self.decomp_way = 'svd'
        with tc.no_grad():
            if if_fast and _center > -0.5:
                p0 = self.length - 1
                p1 = 0
                for n in range(0, self.length - 1):
                    if self.lm[n].numel() == 0:
                        p0 = min(p0, n)
                        p1 = max(p1, n)
                self.central_orthogonalization(p0)
                self.central_orthogonalization(p1+1)
                self.central_orthogonalization(_center)
            else:
                self.central_orthogonalization(0)
                self.central_orthogonalization(self.length-1)
                if _center > 0:
                    self.central_orthogonalization(_center)
            self.decomp_way = _way

    def calculate_entanglement_entropy(self):
        # !!! not fixed for multi-channels
        self.ent = list(range(self.length-1))
        for i in range(0, self.length - 1):
            if self.lm[i].numel() == 0:
                self.ent[i] = -1
            else:
                self.ent[i] = self.entanglement_entropy(
                    self.lm[i].cpu().numpy())

    def mps_zero_grad(self):
        for n in range(self.length):
            if self.tensors[n].grad is not None:
                self.tensors[n].grad.data.zero_()

    @staticmethod
    def entanglement_entropy(lm, tol=1e-20):
        with tc.no_grad():
            lm /= np.linalg.norm(lm)
            lm = lm[lm > tol]
            ent = -2 * (lm ** 2).T.dot(np.log(lm))
        return ent


def left2right_decompose_tensor(tensor, way='qr'):
    s1 = tensor.shape
    dim = min(s1[0]*s1[1], s1[2])
    tensor = tensor.reshape(s1[0] * s1[1], s1[2])
    if way == 1 or way == "svd":
        # Use SVD decomposition
        tensor, lm, v = tc.svd(tensor)
        v = tc.einsum('a,ba->ba', lm[:dim], v[:, :dim])
    else:
        # Use QR decomposition
        tensor, v = tc.qr(tensor)
        lm = tc.zeros(0)
        v = v.transpose(1, 0)
    tensor = tensor[:, :dim].reshape(s1[0], s1[1], dim)
    return tensor, v, dim, lm


def right2left_decompose_tensor(tensor, way='qr', is_full=False):
    s1 = tensor.shape
    tensor = tensor.reshape(s1[0], s1[1]*s1[2])
    dim = min(s1[0], s1[1]*s1[2])
    if way == 1 or way == 'svd':
        # Use SVD decomposition
        tensor, lm, v = tc.svd(tensor.transpose(1, 0))
        v = tc.einsum('a,ba->ba', lm[:dim], v[:, :dim])
    else:
        # Use QR decomposition
        tensor, v = tc.qr(tensor.t())
        lm = tc.zeros(0)
        v = v.transpose(1, 0)
    tensor = tensor[:, :dim].transpose(1, 0).reshape(dim, s1[1], s1[2])
    return tensor, v, dim, lm

