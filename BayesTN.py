import torch as tc
from torch import nn
from TNNalgorithms import parameter_tn_bp_mnist as def_parameters
from TensorNetworkExpasion import TTN_basic, Vectorization, TTN_Pool_2by2to1, \
    TTN_ConvTI_2by2to1, TTN_PoolTI_2by2to1, LinearChannel, Vdim_to_norm
import BasicFun as bf


# ==========================================
def Paras_VL_Bayes_sum1_TN():
    para = def_parameters()
    para['which_TN_set'] = 'BayesTN'
    para['TN'] = 'VL_Bayes_sum1_TN'
    para['batch_size'] = 2000
    para['d'] = 2
    para['chi'] = 2
    para['feature_map'] = 'cos_sin'
    para['mps_init'] = 'No.1'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_Bayes_sum1_TN_BP(TTN_basic):
    """
    All non-TI pool
    High learnability; overfitting
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """

    def __init__(self, para_tn, tensors=None):
        super(VL_Bayes_sum1_TN_BP, self).__init__(num_layers=10)
        theta = 1
        self.f_map = para_tn['feature_map']

        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer1 = TTN_Pool_2by2to1(
            1, 1, 14, 14, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer2 = Vectorization(
            para_tn['chi'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer3 = TTN_Pool_2by2to1(
            1, 1, 7, 7, para_tn['chi'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer4 = Vectorization(
            para_tn['chi'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer5 = TTN_Pool_2by2to1(
            1, 1, 4, 4, para_tn['chi'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer6 = Vectorization(
            para_tn['chi'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer7 = TTN_Pool_2by2to1(
            1, 1, 2, 2, para_tn['chi'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer8 = Vectorization(
            para_tn['chi'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer9 = TTN_Pool_2by2to1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.input_tensors(tensors)

    def forward(self, x):
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            if n in [1, 3, 5, 7]:
                x = tc.sigmoid(x)
            if self.f_map in ['cos_sin', 'normalized_linear']:
                if n in [0, 2, 4, 6, 8]:
                    x = x ** 2
        x = x.squeeze()
        return x


def Paras_VL_Bayes_norm1_TN():
    para = def_parameters()
    para['which_TN_set'] = 'BayesTN'
    para['TN'] = 'VL_Bayes_norm1_TN'
    para['batch_size'] = 2000
    para['d'] = 2
    para['chi'] = 2
    para['feature_map'] = 'cos_sin'
    para['mps_init'] = 'No.1'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_Bayes_norm1_TN_BP(TTN_basic):
    """
    All non-TI pool
    High learnability; overfitting
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_Bayes_norm1_TN_BP, self).__init__(num_layers=10)
        theta = 1
        self.f_map = para_tn['feature_map']
        
        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer1 = TTN_Pool_2by2to1(
            1, 1, 14, 14, para_tn['d'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer2 = Vectorization(
            para_tn['chi'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer3 = TTN_Pool_2by2to1(
            1, 1, 7, 7, para_tn['chi'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer4 = Vectorization(
            para_tn['chi'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer5 = TTN_Pool_2by2to1(
            1, 1, 4, 4, para_tn['chi'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer6 = Vectorization(
            para_tn['chi'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer7 = TTN_Pool_2by2to1(
            1, 1, 2, 2, para_tn['chi'], 1,
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer8 = Vectorization(
            para_tn['chi'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer9 = TTN_Pool_2by2to1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.input_tensors(tensors)

    def forward(self, x):
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            if n in [1, 3, 5, 7]:
                x = tc.sigmoid(x)
        x = x.squeeze()
        return x


# ================================================
def Paras_VL_Bayes_sum1_Vsoftmax_TN():
    para = def_parameters()
    para['which_TN_set'] = 'BayesTN'
    para['TN'] = 'VL_Bayes_sum1_Vsoftmax_TN'
    para['batch_size'] = 2000
    para['d'] = 2
    para['chi'] = 2
    para['feature_map'] = 'cos_sin'
    para['normalize_tensors'] = 'norm2'
    para['update_way'] = 'rotate'

    para['mps_init'] = 'randn'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_Bayes_sum1_Vsoftmax_TN_BP(TTN_basic):
    """
    All non-TI pool
    High learnability; overfitting
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_Bayes_sum1_Vsoftmax_TN_BP, self).__init__(num_layers=6)
        theta = 1
        self.f_map = para_tn['feature_map']
        add_bias = False
        pre_process_tensors = 'square'   # 'normalize', 'softmax'

        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer1 = TTN_Pool_2by2to1(
            1, 1, 14, 14, para_tn['d'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer2 = TTN_Pool_2by2to1(
            1, 1, 7, 7, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer3 = TTN_Pool_2by2to1(
            1, 1, 4, 4, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer4 = TTN_Pool_2by2to1(
            1, 1, 2, 2, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer5 = TTN_Pool_2by2to1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.input_tensors(tensors)

    def forward(self, x):
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            if n == 0:
                x = x ** 2
        x = x.squeeze()
        return x


def Paras_VL_Bayes_sum1_Vsoftmax_TN_adam():
    para = def_parameters()
    para['which_TN_set'] = 'BayesTN'
    para['TN'] = 'VL_Bayes_sum1_Vsoftmax_TN_adam'
    para['batch_size'] = 2000
    para['d'] = 2
    para['chi'] = 2
    para['feature_map'] = 'cos_sin'
    para['normalize_tensors'] = 'norm2'
    para['update_way'] = 'bp'

    para['mps_init'] = 'randn'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_Bayes_sum1_Vsoftmax_TN_adam_BP(TTN_basic):
    """
    All non-TI pool
    High learnability; overfitting
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_Bayes_sum1_Vsoftmax_TN_adam_BP, self).__init__(num_layers=6)
        theta = 1
        self.f_map = para_tn['feature_map']
        add_bias = False
        pre_process_tensors = 'square'   # 'normalize', 'softmax'

        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer1 = TTN_Pool_2by2to1(
            1, 1, 14, 14, para_tn['d'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer2 = TTN_Pool_2by2to1(
            1, 1, 7, 7, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer3 = TTN_Pool_2by2to1(
            1, 1, 4, 4, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer4 = TTN_Pool_2by2to1(
            1, 1, 2, 2, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer5 = TTN_Pool_2by2to1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.input_tensors(tensors)

    def forward(self, x):
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            if n == 0:
                x = x ** 2
        x = x.squeeze()
        return x


def Paras_VL_Bayes_sum1_Vsoftmax_cvTN():
    para = def_parameters()
    para['which_TN_set'] = 'BayesTN'
    para['TN'] = 'VL_Bayes_sum1_Vsoftmax_cvTN'
    para['batch_size'] = 2000
    para['d'] = 2
    para['chi'] = 2
    para['feature_map'] = 'cos_sin'
    para['normalize_tensors'] = 'norm2'
    para['update_way'] = 'rotate'

    para['mps_init'] = 'randn'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_Bayes_sum1_Vsoftmax_cvTN_BP(TTN_basic):
    """
    All non-TI pool
    High learnability; overfitting
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_Bayes_sum1_Vsoftmax_cvTN_BP, self).__init__(num_layers=7)
        theta = 1
        self.f_map = para_tn['feature_map']
        add_bias = False
        pre_process_tensors = 'square'   # 'normalize', 'softmax'

        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer1 = TTN_ConvTI_2by2to1(
            1, 1, para_tn['d'], para_tn['d'], para_tn['device'],
            ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer2 = TTN_Pool_2by2to1(
            1, 1, 14, 14, para_tn['d'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer3 = TTN_Pool_2by2to1(
            1, 1, 7, 7, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer4 = TTN_Pool_2by2to1(
            1, 1, 4, 4, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer5 = TTN_Pool_2by2to1(
            1, 1, 2, 2, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer6 = TTN_Pool_2by2to1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.input_tensors(tensors)

    def forward(self, x):
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            if n == 0:
                x = x ** 2
        x = x.squeeze()
        return x


def Paras_VL_Bayes_norm1_Vnormalize_TN():
    para = def_parameters()
    para['which_TN_set'] = 'BayesTN'
    para['TN'] = 'VL_Bayes_norm1_Vnormalize_TN'
    para['batch_size'] = 2000
    para['d'] = 2
    para['chi'] = 2
    para['feature_map'] = '1MinusSig'
    para['mps_init'] = 'No.1'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 20

    para['it_time'] = 400
    para['lr'] = 1e-3
    return para


class VL_Bayes_norm1_Vnormalize_TN_BP(TTN_basic):
    """
    All non-TI pool
    High learnability; overfitting
                     train_acc   test_acc
    MNIST (d=2)
    f-MNIST(d=2)
    f-MNIST(d=3)
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_Bayes_norm1_Vnormalize_TN_BP, self).__init__(num_layers=6)
        theta = 1
        self.f_map = para_tn['feature_map']

        self.layer0 = Vectorization(
            para_tn['d'], para_tn['feature_map'], para_tn['device'], theta_m=theta)
        self.layer1 = TTN_Pool_2by2to1(
            1, 1, 14, 14, para_tn['d'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer2 = TTN_Pool_2by2to1(
            1, 1, 7, 7, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer3 = TTN_Pool_2by2to1(
            1, 1, 4, 4, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer4 = TTN_Pool_2by2to1(
            1, 1, 2, 2, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.layer5 = TTN_Pool_2by2to1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'])
        self.input_tensors(tensors)

    def forward(self, x):
        for n in range(self.num_layers):
            x = eval('self.layer' + str(n) + '(x)')
            if n in [1, 2, 3, 4]:
                norm = x.norm(dim=2)
                x = tc.einsum('ncdxy,ncxy->ncdxy', [x, 1/(norm + 1e-10)])
        x = x.squeeze()
        return x


