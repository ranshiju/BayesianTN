from TNNalgorithms import tn_multi_classifier_bp_mnist as mps_bp, \
    parameter_tn_bp_mnist as parameter, make_para_consistent


var = [6, 10, 16, 24]
var_name = 'chi'

para, para_fw = parameter()
para['dataset'] = 'mnist'
para['batch_size'] = 6000
para['cut_size'] = [24, 24]
para['img_size'] = [18, 18]
para['it_time'] = 2000
para['lr'] = 1e-4
para['d'] = 4
para['chi'] = 6
para['feature_map'] = 'cos'
para['activate_fun'] = None
para['activate_fun_final'] = None
para['Lagrangian'] = 1e-4
para['check_time'] = 50
para['save_time'] = para['check_time']

for n in range(len(var)):
    para[var_name] = var[n]
    mps_bp(para, para_fw)


