import BasicFun as bf


file = 'data/VL_Bayes_sum1_Vsoftmax_TN/' \
       'VL_BAYES_SUM1_VSOFTMAX_TN_L784_d2_chi18_classes[' \
       '0, 1, 2, 3, 4, 5, 6, 7, 8, 9]_cos_sin_FASHION-MNIST'

info, para = bf.load(file, ['info', 'para'], device='cuda')
print('Check time = ' + str(para['check_time']))
# bf.output_txt(info['train_loss'], 'train_loss')
# bf.output_txt(info['train_acc'], 'train_acc')
# bf.output_txt(info['test_acc'], 'test_acc')
print(info['train_acc'][-1], info['test_acc'][-1])



