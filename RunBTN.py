import CollectedTNN as ctnn
from TNNalgorithms import Generate_by_Bayes_TN, parameters_bayes_generator


file = './data/VL_Bayes_TN/VL_BAYES_TN_L784_d2_chi2_classes[' \
       '0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_cos_sin_FASHION-MNIST'

para_g = parameters_bayes_generator()
para_g['class'] = 0
para_g['num_samples'] = 9
para_g['lr'] = 5e-2
para_g['it_time'] = 500
para_g['check_time'] = 5


Generate_by_Bayes_TN(file, para_g)





