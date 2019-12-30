import torch as tc
from torch import nn
from TNNalgorithms import parameter_tn_bp_mnist as def_parameters
from TensorNetworkExpasion import TTN_basic, Vectorization, TTN_Pool_2by2to1, \
    TTN_Conv_2by2to1, TTN_PoolTI_2by2to1, LinearChannel, Vdim_to_norm
import BasicFun as bf
'''
To name the classes, please begin with 'VL' and end with '_BP'
'''


def Paras_VL_Full_NonTI_Pooling():
    para = def_parameters()
    para['which_TN_set'] = 'ctnn'
    para['TN'] = 'VL_Full_NonTI_Pooling'
    para['batch_size'] = 2000
    para['d'] = 2
    para['feature_map'] = '1Relu'
    para['mps_init'] = 'No.1'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_Full_NonTI_Pooling_BP(TTN_basic):
    """
    All non-TI pool
    High learnability; overfitting
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_Full_NonTI_Pooling_BP, self).__init__(num_layers=10)
        self.channel = para_tn['channel']
        theta = False

        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer1 = TTN_Pool_2by2to1(
            para_tn['channel'], para_tn['channel'], 14, 14, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer2 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer3 = TTN_Pool_2by2to1(
            para_tn['channel'], para_tn['channel'], 7, 7, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer4 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer5 = TTN_Pool_2by2to1(
            para_tn['channel'], para_tn['channel'], 4, 4, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer6 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer7 = TTN_Pool_2by2to1(
            para_tn['channel'], para_tn['channel'], 2, 2, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer8 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer9 = TTN_Pool_2by2to1(
            para_tn['channel'], para_tn['channel'], 1, 1, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.input_tensors(tensors)

    def forward(self, x):
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            # if n in self.activate_layers:
            #     x = tc.sigmoid(x)
        x = tc.tanh(x)
        x = x.squeeze()
        return x


def Paras_VL_test0_Full_NonTI_Pooling():
    para = def_parameters()
    para['which_TN_set'] = 'ctnn'
    para['TN'] = 'VL_test0_Full_NonTI_Pooling'
    para['batch_size'] = 2000
    para['d'] = 3
    para['feature_map'] = '1Relu'
    para['mps_init'] = 'No.1'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_test0_Full_NonTI_Pooling_BP(TTN_basic):
    """
    All non-TI pool
    High learnability; overfitting
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_test0_Full_NonTI_Pooling_BP, self).__init__(num_layers=10)
        self.channel = para_tn['channel']
        theta = False

        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer1 = TTN_Pool_2by2to1(
            para_tn['channel'], para_tn['channel'], 14, 14, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer2 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer3 = TTN_Pool_2by2to1(
            para_tn['channel'], para_tn['channel'], 7, 7, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer4 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer5 = TTN_Pool_2by2to1(
            para_tn['channel'], para_tn['channel'], 4, 4, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer6 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer7 = TTN_Pool_2by2to1(
            para_tn['channel'], para_tn['channel'], 2, 2, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer8 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer9 = TTN_Pool_2by2to1(
            para_tn['channel'], para_tn['channel'], 1, 1, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.input_tensors(tensors)

    def forward(self, x):
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            # if n in self.activate_layers:
            #     x = tc.sigmoid(x)
        x = tc.tanh(x)
        x = x.squeeze()
        return x


# =============================================================
def Paras_VL_Full_TI_Pooling_LargeC():
    para = def_parameters()
    para['which_TN_set'] = 'ctnn'
    para['TN'] = 'VL_Full_TI_Pooling_LargeC'
    para['batch_size'] = 2000
    para['d'] = 2
    para['feature_map'] = '1Relu'
    para['mps_init'] = 'No.1'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20
    # para['optimizer'] = 'SGD'

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_Full_TI_Pooling_LargeC_BP(TTN_basic):
    """
    PPP... (all TI) + large channel
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_Full_TI_Pooling_LargeC_BP, self).__init__(num_layers=11)
        self.channel = para_tn['channel']
        theta_m = False

        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer1 = TTN_PoolTI_2by2to1(
            1, 5, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 14, 14
        self.layer2 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer3 = TTN_PoolTI_2by2to1(
            5, 20, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 7, 7
        self.layer4 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer5 = TTN_PoolTI_2by2to1(
            20, 100, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 4, 4
        self.layer6 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer7 = TTN_PoolTI_2by2to1(
            100, 200, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer8 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer9 = TTN_PoolTI_2by2to1(
            200, 400, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'], out_dims=4)
        self.layer10 = nn.Linear(400, self.channel).to(para_tn['device'])

        layer_dict = {'layer0.0': 'layer1',
                      'layer1.0': 'layer3',
                      'layer2.0': 'layer5',
                      'layer3.0': 'layer7',
                      'layer4.0': 'layer9'}
        self.input_tensors_TN_from_NN_Conv2D(
            './data/VL_NN_mimic0/VL_NN_MIMIC0_L784_d2_chi2_classes['
            '0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_RelSig_FASHION-MNIST',
            layer_dict, para_tn['device'])
        self.input_tensors_NN_from_NN(
            './data/VL_NN_mimic0/VL_NN_MIMIC0_L784_d2_chi2_classes['
            '0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_RelSig_FASHION-MNIST',
            {'layer5.0': 'layer10'}, para_tn['device'])
        # self.input_tensors(tensors)
        # self.manual_set_paras_groups(para_tn['lr'], [['layer1', 'layer3'], []])

    def forward(self, x):
        # x.shape = num, channel, d, lx, ly
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            # if n in [1, 3, 5, 7, 9]:
            #     x = tc.sigmoid(x)
            if n in [9]:
                # x = tc.sigmoid(x)
                x = nn.ReLU(inplace=True)(x)
                x = x.view(x.shape[0], -1)
            if n in [10]:
                x = nn.Tanh()(x)
        x = x.view(x.shape[0], -1)
        # x = x.squeeze()
        return x


# =============================================================
def Paras_VL_NN():
    para = def_parameters()
    para['which_TN_set'] = 'cnn'
    para['TN'] = 'VL_NN'
    para['batch_size'] = 1000
    para['d'] = 2
    para['feature_map'] = 'RelSig'
    para['mps_init'] = 'rand'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_NN_BP(TTN_basic):

    def __init__(self, para_tn, tensors=None):
        super(VL_NN_BP, self).__init__(num_layers=11)
        self.channel = para_tn['channel']

        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2),  # 16, 26 ,26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)).to(para_tn['device'])

        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # 32, 24, 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)).to(para_tn['device'])  # 32, 12,12     (24-2) /2 +1

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  # 64,10,10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)).to(para_tn['device'])

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  # 128,8,8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)).to(para_tn['device'])  # 128, 4,4

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.channel),
            nn.Tanh()).to(para_tn['device'])
        self.input_tensors(tensors)

    def forward(self, x):
        import time
        x = self.pre_process_data_dims(x)
        # t0 = time.time()
        # print(x.shape)
        x = self.layer0(x)
        # print(0)
        # print(time.time() - t0)
        # t0 = time.time()
        x = self.layer1(x)
        # print(1)
        # print(time.time() - t0)
        # t0 = time.time()
        x = self.layer2(x)
        # print(2)
        # print(time.time() - t0)
        # t0 = time.time()
        x = self.layer3(x)
        # print(3)
        # print(time.time() - t0)
        # t0 = time.time()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # print(4)
        # print(time.time() - t0)
        # t0 = time.time()
        return x


# =============================================================
def Paras_VL_NN_TIPoolingC400():
    para = def_parameters()
    para['which_TN_set'] = 'ctnn'
    para['TN'] = 'VL_NN_TIPoolingC400'
    para['batch_size'] = 1000
    para['d'] = 2
    para['feature_map'] = 'RelSig'
    para['mps_init'] = 'rand'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_NN_TIPoolingC400_BP(TTN_basic):

    def __init__(self, para_tn, tensors=None):
        super(VL_NN_TIPoolingC400_BP, self).__init__(num_layers=11)
        self.channel = para_tn['channel']
        a_fun = 'nn.ReLU(inplace=True)'
        # a_fun = 'nn.Sigmoid()'

        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 5, kernel_size=2, stride=2),  # 14 ,14
            eval(a_fun)).to(para_tn['device'])
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 20, kernel_size=2, stride=2),  # 7, 7
            eval(a_fun)).to(para_tn['device'])
        self.layer2 = nn.Sequential(
            nn.Conv2d(20, 100, kernel_size=2, stride=2),  # 4, 4
            eval(a_fun)).to(para_tn['device'])
        self.layer3 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=2, stride=2),  # 2, 2
            eval(a_fun)).to(para_tn['device'])
        self.layer4 = nn.Sequential(
            nn.Conv2d(200, 400, kernel_size=2, stride=2),
            eval(a_fun)).to(para_tn['device'])
        self.layer5 = nn.Sequential(
            nn.Linear(400, para_tn['channel']),
            nn.Tanh()).to(para_tn['device'])
        self.input_tensors(tensors)

    def forward(self, x):
        import time
        x = self.pre_process_data_dims(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = bf.pad_x_copy_one_line(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        return x


# ----------------------------------------
def Paras_VL_TIPoolingC400_1():
    para = def_parameters()
    para['which_TN_set'] = 'ctnn'
    para['TN'] = 'VL_TIPoolingC400_1'
    para['batch_size'] = 2000
    para['d'] = 2
    para['feature_map'] = '1Relu'
    para['mps_init'] = 'No.1'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20
    # para['optimizer'] = 'SGD'

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_TIPoolingC400_1_BP(TTN_basic):
    """
    PPP... (all TI) + large channel
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_TIPoolingC400_1_BP, self).__init__(num_layers=11)
        self.channel = para_tn['channel']
        theta_m = False

        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer1 = TTN_PoolTI_2by2to1(
            1, 5, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 14, 14
        self.layer2 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer3 = TTN_PoolTI_2by2to1(
            5, 20, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 7, 7
        self.layer4 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer5 = TTN_PoolTI_2by2to1(
            20, 100, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 4, 4
        self.layer6 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer7 = TTN_PoolTI_2by2to1(
            100, 200, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer8 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer9 = TTN_PoolTI_2by2to1(
            200, 400, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'], out_dims=4)
        self.layer10 = nn.Linear(400, self.channel).to(para_tn['device'])

        if tensors is None:
            layer_dict = {'layer0.0': 'layer1',
                          'layer1.0': 'layer3',
                          'layer2.0': 'layer5',
                          'layer3.0': 'layer7',
                          'layer4.0': 'layer9'}
            self.input_tensors_TN_from_NN_Conv2D(
                './data/VL_NN_TIPoolingC400/VL_NN_TIPOOLINGC400_L784_d2_chi2_classes['
                '0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_RelSig_FASHION-MNIST',
                layer_dict, para_tn['device'])
            self.input_tensors_NN_from_NN(
                './data/VL_NN_TIPoolingC400/VL_NN_TIPOOLINGC400_L784_d2_chi2_classes['
                '0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_RelSig_FASHION-MNIST',
                {'layer5.0': 'layer10'}, para_tn['device'])
        else:
            self.input_tensors(tensors)
        self.manual_set_paras_groups(para_tn['lr'], [
            ['layer1'], ['layer3'], ['layer5'], ['layer7'], ['layer9'], ['layer10']])
        # input()

    def forward(self, x):
        # x.shape = num, channel, d, lx, ly
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            # if n in [1, 3, 5, 7, 9]:
            #     x = tc.sigmoid(x)
            if n in [9]:
                # x = tc.sigmoid(x)
                x = nn.ReLU(inplace=True)(x)
                x = x.view(x.shape[0], -1)
            if n in [10]:
                x = nn.Tanh()(x)
        x = x.view(x.shape[0], -1)
        # x = x.squeeze()
        return x


# ----------------------------------------
def Paras_VL_TIPoolingC400_2():
    para = def_parameters()
    para['which_TN_set'] = 'ctnn'
    para['TN'] = 'VL_TIPoolingC400_2'
    para['batch_size'] = 2000
    para['d'] = 3
    para['feature_map'] = '1RelSig'
    para['mps_init'] = 'No.1'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_TIPoolingC400_2_BP(TTN_basic):
    """
    PPP... (all TI) + large channel
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_TIPoolingC400_2_BP, self).__init__(num_layers=11)
        self.channel = para_tn['channel']
        theta_m = False

        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer1 = TTN_PoolTI_2by2to1(
            1, 5, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 14, 14
        self.layer2 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer3 = TTN_PoolTI_2by2to1(
            5, 20, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 7, 7
        self.layer4 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer5 = TTN_PoolTI_2by2to1(
            20, 100, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 4, 4
        self.layer6 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer7 = TTN_PoolTI_2by2to1(
            100, 200, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer8 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer9 = TTN_PoolTI_2by2to1(
            200, 400, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'], out_dims=4)
        self.layer10 = nn.Linear(400, self.channel).to(para_tn['device'])

        if tensors is None:
            layer_dict = {'layer0.0': 'layer1',
                          'layer1.0': 'layer3',
                          'layer2.0': 'layer5',
                          'layer3.0': 'layer7',
                          'layer4.0': 'layer9'}
            self.input_tensors_TN_from_NN_Conv2D(
                './data/VL_NN_TIPoolingC400/VL_NN_TIPOOLINGC400_L784_d2_chi2_classes['
                '0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_RelSig_FASHION-MNIST',
                layer_dict, para_tn['device'])
            self.input_tensors_NN_from_NN(
                './data/VL_NN_TIPoolingC400/VL_NN_TIPOOLINGC400_L784_d2_chi2_classes['
                '0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_RelSig_FASHION-MNIST',
                {'layer5.0': 'layer10'}, para_tn['device'])
        else:
            self.input_tensors(tensors)
        self.manual_set_paras_groups(para_tn['lr'], [
            ['layer1'], ['layer3'], ['layer5'], ['layer7'], ['layer9'], ['layer10']])

    def forward(self, x):
        # x.shape = num, channel, d, lx, ly
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            # if n in [1, 3, 5, 7, 9]:
            #     x = tc.sigmoid(x)
            if n in [9]:
                # x = tc.sigmoid(x)
                x = nn.ReLU(inplace=True)(x)
                x = x.view(x.shape[0], -1)
            if n in [10]:
                x = nn.Tanh()(x)
        x = x.view(x.shape[0], -1)
        # x = x.squeeze()
        return x


# ----------------------------------------
def Paras_VL_TIPoolingC400_3():
    para = def_parameters()
    para['which_TN_set'] = 'ctnn'
    para['TN'] = 'VL_TIPoolingC400_3'
    para['batch_size'] = 2000
    para['d'] = 4
    para['feature_map'] = '1RelSigCos'
    para['mps_init'] = 'No.1'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_TIPoolingC400_3_BP(TTN_basic):
    """
    PPP... (all TI) + large channel
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_TIPoolingC400_3_BP, self).__init__(num_layers=11)
        self.channel = para_tn['channel']
        theta_m = False

        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer1 = TTN_PoolTI_2by2to1(
            1, 5, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 14, 14
        self.layer2 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer3 = TTN_PoolTI_2by2to1(
            5, 20, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 7, 7
        self.layer4 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer5 = TTN_PoolTI_2by2to1(
            20, 100, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 4, 4
        self.layer6 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer7 = TTN_PoolTI_2by2to1(
            100, 200, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer8 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta_m)
        self.layer9 = TTN_PoolTI_2by2to1(
            200, 400, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'], out_dims=4)
        self.layer10 = nn.Linear(400, self.channel).to(para_tn['device'])

        if tensors is None:
            layer_dict = {'layer0.0': 'layer1',
                          'layer1.0': 'layer3',
                          'layer2.0': 'layer5',
                          'layer3.0': 'layer7',
                          'layer4.0': 'layer9'}
            self.input_tensors_TN_from_NN_Conv2D(
                './data/VL_NN_TIPoolingC400/VL_NN_TIPOOLINGC400_L784_d2_chi2_classes['
                '0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_RelSig_FASHION-MNIST',
                layer_dict, para_tn['device'])
            self.input_tensors_NN_from_NN(
                './data/VL_NN_TIPoolingC400/VL_NN_TIPOOLINGC400_L784_d2_chi2_classes['
                '0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_RelSig_FASHION-MNIST',
                {'layer5.0': 'layer10'}, para_tn['device'])
        else:
            self.input_tensors(tensors)
        self.manual_set_paras_groups(para_tn['lr'], [
            ['layer1'], ['layer3'], ['layer5'], ['layer7'], ['layer9'], ['layer10']])

    def forward(self, x):
        # x.shape = num, channel, d, lx, ly
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            # if n in [1, 3, 5, 7, 9]:
            #     x = tc.sigmoid(x)
            if n in [9]:
                # x = tc.sigmoid(x)
                x = nn.ReLU(inplace=True)(x)
                x = x.view(x.shape[0], -1)
            if n in [10]:
                x = nn.Tanh()(x)
        x = x.view(x.shape[0], -1)
        # x = x.squeeze()
        return x


# =============================================================
def Paras_VL_CNN1():
    para = def_parameters()
    para['which_TN_set'] = 'ctnn'
    para['TN'] = 'VL_CNN1'
    para['batch_size'] = 1000
    para['d'] = 2
    para['feature_map'] = '1Rel'
    para['mps_init'] = 'No.1'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_CNN1_BP(TTN_basic):

    def __init__(self, para_tn, tensors=None):
        super(VL_CNN1_BP, self).__init__(num_layers=8)
        self.channel = para_tn['channel']

        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),  # 24, 24
            nn.ReLU(inplace=True)).to(para_tn['device'])

        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5),  # 20, 20
            nn.ReLU(inplace=True)).to(para_tn['device'])

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),  # 16, 16
            nn.ReLU(inplace=True)).to(para_tn['device'])

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)).to(para_tn['device'])  # 8, 8

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)).to(para_tn['device'])  # 4, 4

        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)).to(para_tn['device'])  # 2, 2

        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)).to(para_tn['device'])  # 1, 1

        self.layer7 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.channel),
            nn.Tanh()).to(para_tn['device'])

        self.input_tensors(tensors)

    def forward(self, x):
        # x.shape = num, channel, d, lx, ly
        x = self.pre_process_data_dims(x)
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            if n == 6:
                x = x.view(x.shape[0], -1)
        x = x.view(x.shape[0], -1)
        return x


def Paras_VL_CNN1TN1():
    para = def_parameters()
    para['which_TN_set'] = 'ctnn'
    para['TN'] = 'VL_CNN1TN1'
    para['batch_size'] = 1000
    para['d'] = 2
    para['feature_map'] = '1Relu'
    para['mps_init'] = 'No.1'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_CNN1TN1_BP(TTN_basic):

    def __init__(self, para_tn, tensors=None):
        super(VL_CNN1TN1_BP, self).__init__(num_layers=12)
        self.channel = para_tn['channel']
        theta = False

        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),  # 24, 24
            nn.ReLU(inplace=True)).to(para_tn['device'])

        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5),  # 20, 20
            nn.ReLU(inplace=True)).to(para_tn['device'])

        self.layer2 = nn.Sequential(  # 16, 16
            nn.Conv2d(16, 32, kernel_size=5)).to(para_tn['device'])

        self.layer3 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)

        self.layer4 = TTN_PoolTI_2by2to1(
            32, 64, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 8, 8

        self.layer5 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)

        self.layer6 = TTN_PoolTI_2by2to1(
            64, 128, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 4, 4

        self.layer7 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)

        self.layer8 = TTN_PoolTI_2by2to1(
            128, 256, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 2, 2

        self.layer9 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)

        self.layer10 = TTN_PoolTI_2by2to1(
            256, 512, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])  # 1, 1

        self.layer11 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, self.channel),
            nn.Tanh()).to(para_tn['device'])

        if tensors is None:
            layer_dict = {'layer3.0': 'layer4',
                          'layer4.0': 'layer6',
                          'layer5.0': 'layer8',
                          'layer6.0': 'layer10'}
            self.input_tensors_TN_from_NN_Conv2D(
                './data/VL_CNN1/VL_CNN1_L784_d2_chi2_classes[0, 1, 2, 3, 4, '
                '5, 6, 7, 8, 9]_1Rel_FASHION-MNIST',
                layer_dict, para_tn['device'])
            layer_dict1 = {'layer0.0': 'layer0.0',
                           'layer1.0': 'layer1.0',
                           'layer2.0': 'layer2.0',
                           'layer7.0': 'layer11.0',
                           'layer7.2': 'layer11.2',
                           'layer7.4': 'layer11.4'}
            self.input_tensors_NN_from_NN(
                './data/VL_CNN1/VL_CNN1_L784_d2_chi2_classes[0, 1, 2, 3, 4, '
                '5, 6, 7, 8, 9]_1Rel_FASHION-MNIST',
                layer_dict1, para_tn['device'])
        else:
            self.input_tensors(tensors)
        self.manual_set_paras_groups(para_tn['lr'], [
            ['layer0', 'layer1', 'layer2'], ['layer4'], ['layer6'],
            ['layer8'], ['layer10'], ['layer11']])

    def forward(self, x):
        # x.shape = num, channel, d, lx, ly
        x = self.pre_process_data_dims(x)
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            if n == 10:
                x = nn.ReLU(inplace=True)(x)
                x = x.view(x.shape[0], -1)
        x = x.view(x.shape[0], -1)
        return x

