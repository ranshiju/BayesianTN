import os
import PlotFun as pf
import TensorNetworkExpasion as tne


exp = 'MPS_L784_d2_chi6_classes[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_taylor_FASHION-MNIST'


mps, info = tne.load_tensor_network(os.path.join('./data/MPS', exp), 'cuda')

# pf.plot(info['train_acc'])
ent = mps.calculate_ent_onebody()
tne.plot_ent_2d(ent, (28, 28))
