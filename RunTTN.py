import CollectedTNN as ctnn
from TNNalgorithms import tn_multi_classifier_bp_mnist as mps_bp


TNN = 'VL_Bayes_sum1_Vsoftmax_TN_adam'
exec('from BayesTN import Paras_' + TNN)
para = eval('Paras_' + TNN + '()')

para['d'] = 2
para['chi'] = 2
para['dataset'] = 'fashion-mnist'
para['batch_size'] = 2000

para['check_time'] = 1
para['save_time'] = 20
para['it_time'] = 400
para['lr'] = 1e-3
para['Lagrangian'] = None

mps_bp(para)

# para['it_time'] = 200
# para['lr'] = 5e-4
# mps_bp(para)

