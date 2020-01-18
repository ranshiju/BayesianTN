import CollectedTNN as ctnn
from TNNalgorithms import tn_multi_classifier_bp_mnist as mps_bp


# TNN = 'VL_Bayes_sum1_Vsoftmax_TN_adam'
# exec('from BayesTN import Paras_' + TNN)
# para = eval('Paras_' + TNN + '()')

# Important
from BayesTN import Paras_VL_Bayes_sum1_Vsoftmax_TN_2to1
para = Paras_VL_Bayes_sum1_Vsoftmax_TN_2to1()

para['d'] = 2
para['chi'] = 2
para['dataset'] = 'fashion-mnist'
para['batch_size'] = 6000

para['check_time'] = 1
para['save_time'] = 5
para['it_time'] = 400
para['lr'] = 1e-2
para['Lagrangian'] = None
para['if_load'] = False
mps_bp(para)

# para['it_time'] = 200
# para['lr'] = 5e-4
# mps_bp(para)

