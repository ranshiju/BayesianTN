from TensorNetworkExpasion import load_mps_and_test_mnist as test_mnist
import PlotFun as pf


data_path = './data/MPS/'
dataset = 'MNIST'
d = [2, 4, 6]

data_exp1 = 'MPS_L784_d'
data_exp2 = '_chi6_classes[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_taylor_'

acc = list()
for n in range(len(d)):
    data_exp = data_path, data_exp1 + str(d[n]) + data_exp2 + dataset
    acc.append(test_mnist(data_exp)[1])
pf.plot(d, acc)




