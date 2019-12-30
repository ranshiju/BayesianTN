from os.path import join
from TensorNetworkExpasion import load_mps_and_test_mnist


path = './data/VLTTN2_saved'
file = 'VLTTN2_L784_d3_chi2_classes[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_cos_sin_FASHION-MNIST'

load_mps_and_test_mnist(join(path, file))

