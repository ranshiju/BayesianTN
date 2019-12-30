import os
import sys
import time
import copy

import torch as tc
from termcolor import cprint

import BasicFun as bf
import TensorNetworkExpasion as tne
from pytorch_gpu_memory.gpu_memory_log import gpu_memory_log

dev = bf.choose_device()
dtp = tc.float32


def parameter_tn_bp_mnist():
    para = dict()
    para['project'] = 'TNMLbp'
    para['which_TN_set'] = 'tne'  # 'tne' or 'ctnn'
    para['TN'] = 'MPS'

    para['dataset'] = 'fashion-mnist'
    para['classes'] = list(range(10))
    para['num_samples'] = ['all'] * para['classes'].__len__()
    para['batch_size'] = 3000

    para['binary_imgs'] = False
    para['cut_size'] = [28, 28]
    para['img_size'] = [28, 28]
    # to feature cut-off; not usable yet
    para['update_f_index'] = False
    para['tol_cut_f'] = 1e-12

    para['it_time'] = 200
    para['lr'] = 1e-5
    para['d'] = 2
    para['chi'] = 2

    para['linear_gauss_noise'] = None
    para['pre_normalize_mps'] = 1
    para['normalize_mps'] = False
    para['optimizer'] = 'Adam'
    para['mps_init'] = 'No.1'
    para['feature_map'] = 'taylor'
    para['feature_theta'] = 1
    para['activate_fun'] = None
    para['activate_fun_final'] = None
    para['Lagrangian'] = None
    para['Lagrangian_way'] = 0
    para['norm_p'] = 1
    para['loss_func'] = 'CrossEntropyLoss'  # MSELoss, CrossEntropyLoss, NLLLoss

    para['check_time'] = 2
    para['save_time'] = 20
    para['if_test'] = True
    para['if_load'] = True
    para['if_load_smaller_chi'] = True
    para['clear_history'] = False
    para['normalize_tensors'] = None
    para['update_way'] = 'bp'
    para['multi_gpu_parallel'] = False

    para['log_name'] = 'record'
    para['device'] = 'cuda'

    para = make_para_consistent(para)
    return para


def make_para_consistent(para):
    if 'TN' not in para:
        para['TN'] = 'MPS'
    if 'norm_p' not in para:
        para['norm_p'] = 1
    if 'binary_imgs' not in para:
        para['binary_imgs'] = False
    if para['TN'] != 'MPS':
        para['normalize_mps'] = False
        para['activate_fun'] = None
        para['activate_fun_final'] = None
    para['data_path'] = './data/' + para['TN'] + '/'
    if para['feature_map'] == 'fold_2d_order1':
        para['img_size'] = [round(para['img_size'][0]/2),
                            round(para['img_size'][1]/2)]
    if para['feature_map'].lower() in ['normalized_linear',
                                       'relsig', 'tansig', 'vsigmoid']:
        if para['d'] != 2:
            bf.warning('Warning: Inconsistent para[\'d\']=%g to '
                       'feature map. Please check...' % para['d'])
            para['d'] = 2
    if para['feature_map'].lower() == 'reltansig':
        if para['d'] != 3:
            bf.warning('Warning: Inconsistent para[\'d\']=%g to '
                       'feature map. Please check...' % para['d'])
            para['d'] = 3
    para['length'] = para['img_size'][0] * para['img_size'][1]
    if 'feature_index' not in para:
        para['feature_index'] = None
    elif para['feature_index'] is not None:
        if len(para['feature_index']) > para['length']:
            bf.warning('Error: length > len(feature_index).')
            sys.exit(1)
        elif max(para['feature_index']) > (para['length'] - 1):
            bf.warning('Error: feature_index.max() > len(feature_index).')
            sys.exit(1)
        else:
            para['length'] = len(para['feature_index'])
    para['channel'] = len(para['classes'])
    para['data_exp'] = data_exp_to_save_mps(para)
    if (para['device'] != 'cpu') and (not tc.cuda.is_available()):
        para['device'] = 'cpu'
        bf.warning('Cuda is not available in the device...')
        bf.warning('Changed to \'cpu\' instead...')
    return para


def data_exp_to_save_mps(para):
    exp = para['TN'].upper() + '_L' + str(para['length']) + '_d' + str(para['d']) + '_chi' + \
               str(para['chi']) + '_classes' + str(para['classes']) + '_' + \
               para['feature_map'] + '_' + para['dataset'].upper()
    if para['dataset'].lower() in ['mnist', 'fashion-mnist', 'fashionmnist']:
        if (para['cut_size'][0] != 28) or (para['cut_size'][1] != 28):
            exp += ('_size' + str(para['cut_size']))
        if (para['img_size'][0] != 28) or (para['img_size'][1] != 28):
            exp += str(para['img_size'])
    elif para['dataset'].lower() in ['cifar10', 'cifar-10']:
        if (para['cut_size'][0] != 32) or (para['cut_size'][1] != 32):
            exp += ('_size' + str(para['cut_size']))
        if (para['img_size'][0] != 32) or (para['img_size'][1] != 32):
            exp += str(para['img_size'])
    if 'feature_index' in para:
        if para['feature_index'] is not None:
            exp += '_FindexedNum' + str(len(para['feature_index']))
    if para['binary_imgs']:
        exp += 'binary'
    return exp


def load_saved_tn_smaller_chi_d(para, path1=None):
    if para['if_load']:
        path = './data/' + para['TN'] + '/'
        exp = data_exp_to_save_mps(para)
        mps_file = os.path.join(path, exp)
        if os.path.isfile(mps_file):
            message = 'Load existing ' + para['TN'] + ' data...'
            mps, info, _ = tne.load_tensor_network(mps_file, para)
            return mps, info, message
        elif para['if_load_smaller_chi']:
            if path1 is None:
                path1 = './data/' + para['TN'] + '_saved/'
            chi0 = para['chi']
            d0 = para['d']
            for d in range(d0, 1, -1):
                for chi in range(chi0, 1, -1):
                    para['d'] = d
                    para['chi'] = chi
                    exp = data_exp_to_save_mps(para)
                    mps_file = os.path.join(path1, exp)
                    if os.path.isfile(mps_file):
                        message = 'Load existing ' + para['TN'] + ' with (d, chi) = ' + \
                                  str((para['d'], para['chi']))
                        para['chi'], para['d'] = chi0, d0
                        mps, info, _ = tne.load_tensor_network(
                            mps_file, para)
                        return mps, info, message
            message = 'No existing smaller-chi/d ' + \
                      para['TN'] + ' found...\n ' \
                                   'Create new ' + para['TN'] + ' data ...'
            para['chi'], para['d'] = chi0, d0
            return None, None, message
        else:
            message = 'No existing ' + para['TN'] + ' found...\n Create new ' + \
                      para['TN'] + ' data ...'
            return None, None, message
    else:
        return None, None, 'Create new ' + para['TN'] + ' data ...'


def tn_multi_classifier_bp_mnist(para=None):
    logger = bf.logger(para['log_name']+'.log', level='info')
    log = logger.logger.info
    t0 = time.time()
    if para is None:
        para = parameter_tn_bp_mnist()
    para = make_para_consistent(para)
    log('=' * 15)
    log('Using device: ' + para['device'])
    log('=' * 15)
    bf.print_dict(para)

    labels2mat = (para['loss_func'] == 'MSELoss')
    if para['TN'] == 'MPS':
        data_dim = 2
    else:
        data_dim = 5
    train_loader, test_loader = bf.load_mnist_and_batch(
        para['dataset'], para['classes'], para['num_samples'], None, para['batch_size'],
        data_dim=data_dim, labels2mat=labels2mat, channel=len(para['classes']),
        project_name=para['project'], dev=para['device'])

    train_loader, train_num_tot = tne.pre_process_dataset(
        train_loader, para, para['device'])
    test_loader, test_num_tot = tne.pre_process_dataset(
        test_loader, para, para['device'])

    num_batch_train = len(train_loader)
    log('Num of training samples:\t' + str(train_num_tot))
    log('Num of testing samples:\t' + str(test_num_tot))
    log('Num of training batches:\t' + str(num_batch_train))
    log('Num of features:\t' + str(para['length']))
    log('Dataset finish processed...')

    loss_func = eval('tc.nn.' + para['loss_func'] + '()')
    activate_func = None
    if para['activate_fun_final'] is not None:
        activate_func = eval('tc.' + para['activate_fun_final'])

    tn, info, message = load_saved_tn_smaller_chi_d(para)
    log(message)
    flag_new_tn = (tn is None)
    if flag_new_tn:
        exec('from ' + para['which_TN_set'] + ' import ' + para['TN'] + '_BP')
        tn = eval(para['TN'] + '_BP(para)')
        info = dict()
        info['train_acc'] = list()
        info['train_loss'] = list()
        info['test_acc'] = list()
        info['norm_coeff'] = list()
    if para['normalize_tensors'] is not None:
        tn.normalize_all_tensors(para['normalize_tensors'])
    # if flag_new_tn:
    #     tn.pre_normalize(train_loader[0][0].to(
    #         para['device']), para['activate_fun'], para['pre_normalize_mps'])

    nc = tne.test_accuracy_mnist(tn, test_loader, para)
    log('Initially, we have test acc = ' + str(nc / test_num_tot))

    # for x in tn.state_dict():
    #     print(x)
    # print(tn.state_dict()['layer5.0.weight'].shape)
    # print(tn.state_dict()['layer5.0.bias'].shape)
    # input()

    optimizer = tne.assign_optimizers(para['optimizer'], tn.paras_group, 'tn')
    if isinstance(optimizer, str):
        optimizer = eval(optimizer)

    log('Start training...')
    coeff_norm = 0
    if para['if_test']:
        titles = 'Epoch \t train_loss \t train_acc \t test_acc \t norm_coeff'
    else:
        titles = 'Epoch \t train_loss \t train_acc \t norm_coeff'
    log(titles)

    for t in range(para['it_time']):
        if type(optimizer) in [list, tuple]:
            n_optm = (t % len(optimizer))
            # n_optm = 0
            tn = tne.control_require_grad(tn, n_optm)
        t_loop = time.time()
        train_loss = 0
        nc = 0
        if (num_batch_train > 1) and (t > 0):
            train_loader = bf.re_batch_data_loader(train_loader)
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(para['device']), labels.to(para['device'])
            if 'linear_gauss_noise' in para:
                imgs = tne.add_linear_gaussian_noise(imgs, para['linear_gauss_noise'])

            # time_tmp = time.time()
            y = tn(imgs)
            # print(y.sum(dim=1))
            # print(' forward ')
            # print(time.time() - time_tmp)

            if para['normalize_mps']:
                norm_t = tn.calculate_norm_mps()
                y = tc.einsum('nc,c->nc', y, 1 / (norm_t + 1e-12))
            if para['activate_fun_final'] is not None:
                y = activate_func(y)
            loss = loss_func(y, labels)
            with tc.no_grad():
                train_loss += loss.data.item()
            if para['Lagrangian'] is not None:
                loss1 = tn.calculate_norm_tensors(
                    para['Lagrangian_way'], p=para['norm_p'])
                with tc.no_grad():
                    coeff_norm = loss1.data.item()
                loss = loss + loss1 * para['Lagrangian'] / num_batch_train
            elif ((t + 1) % para['check_time']) == 0:
                with tc.no_grad():
                    coeff_norm = tn.calculate_norm_tensors(
                        0, p=para['norm_p']).data.item()

            # print(' backward ')
            # time_tmp = time.time()
            loss.backward()

            if para['update_way'] == 'bp':
                if type(optimizer) in [list, tuple]:
                    optimizer[n_optm].step()
                    optimizer[n_optm].zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()
                tn.normalize_all_tensors(para['normalize_tensors'])
            else:  # rotate; tensors should ne normalized
                for x in tn.parameters():
                    s = x.shape
                    # put grad in tangent space
                    inner = tc.einsum('ac,ac->a', [
                        x.data.view(-1, s[-1]), x.grad.data.view(-1, s[-1])])
                    grad = x.grad.data.view(-1, s[-1]) - tc.einsum(
                        'a,ab->ab', [inner, x.data.view(-1, s[-1])])
                    # normalize grad
                    norm = grad.norm(dim=1, p=2) + 1e-12
                    grad = tc.einsum('ab,a->ab', [grad, 1 / norm])
                    # update x
                    x.data -= para['lr'] * grad.view(s)
                    # normalize x
                    norm = x.data.norm(dim=x.ndimension() - 1, p=2) + 1e-12
                    x.data = tc.einsum('ab,a->ab', [
                        x.data.view(-1, s[-1]), 1 / norm.view(-1, )])
                    x.data = x.data.view(s)
                    x.grad.data.zero_()
            # for x in tn.state_dict():
            #     tn.state_dict()[x].data = \
            #         tn.state_dict()[x].data - para['lr'] * tn.state_dict()[
            #         x].grad / (tn.state_dict()[x].grad.norm(p=1) + 1e-12)
            #     tn.state_dict()[x].grad.data.zero_()

            # for x in tn.parameters():
            #     s = x.shape
            #     norm = x.grad.norm(dim=x.ndimension() - 1, p=1) + 1e-10
            #     grad = tc.einsum('ab,a->ab', [x.grad.view(-1, s[-1]), 1 / norm.view(-1, )])
            #     x.data -= para['lr'] * grad.view(s)
            #
            #     norm = x.data.norm(dim=x.ndimension()-1, p=1) + 1e-10
            #     x.data = tc.einsum('ab,a->ab', [x.data.view(-1, s[-1]), 1/norm.view(-1,)])
            #     x.data = x.data.view(s)
            #     x.grad.data.zero_()

            # print(time.time() - time_tmp)
            # if para['normalize_tensors_sum1'] is not None:
            #     tn.normalize_all_tensors(para['normalize_tensors_sum1'])

            if ((t + 1) % para['check_time']) == 0:
                nc0, _ = tne.num_correct(labels, y.data)
                nc += nc0
        if ((t + 1) % para['check_time']) == 0:
            info['train_acc'].append(nc / train_num_tot)
            info['train_loss'].append(train_loss)
            info['norm_coeff'].append(coeff_norm)
            message = str(t + 1) + ': '
            message += '\t %.6g' % info['train_loss'][-1]
            message += '\t %.6g' % info['train_acc'][-1]
            if para['if_test']:
                nc = tne.test_accuracy_mnist(
                    tn, test_loader, para)
                info['test_acc'].append(nc / test_num_tot)
                message += '\t %.6g' % info['test_acc'][-1]
            message += '\t %.6g' % info['norm_coeff'][-1]
            log(message)
        if ((t+1) % para['save_time']) == 0:
            if (train_loss == float('nan')) or (train_loss == float('inf')):
                cprint('DO NOT save MPS since NAN/INF appears', color='red')
                sys.exit(1)
            else:
                info['time_1loop'] = time.time() - t_loop
                tne.save_tensor_network(tn, para, info,
                                        para['data_path'], para['data_exp'])
                log('MPS saved: time cost per epoch = ' + str(info['time_1loop']))
                log(titles)
    info['time_tot'] = time.time() - t0
    log('Total time cost = ' + str(info['time_tot']))
    return para['data_path'], para['data_exp']


def parameters_bayes_generator():
    para = dict()
    para['class'] = 0
    para['num_samples'] = 4
    para['lr'] = 1e-3
    para['it_time'] = 100
    para['check_time'] = 5

    para['which_TN_set'] = 'BayesTN'
    para['TN'] = 'VL_Bayes_TN'
    para['optimizer'] = 'Adam'
    para['device'] = 'cuda'

    para['img_file_name'] = 'img'
    return para


def Generate_by_Bayes_TN(file, para_g=None):
    from torch import nn
    if para_g is None:
        para_g = parameters_bayes_generator()
    tn, info, para = tne.load_tensor_network(file, para_g)
    for x in tn.parameters():
        x.requires_grad = False

    labels2mat = (para['loss_func'] == 'MSELoss')
    if para['TN'] == 'MPS':
        data_dim = 2
    else:
        data_dim = 5
    train_loader, test_loader = bf.load_mnist_and_batch(
        para['dataset'], para['classes'], para['num_samples'], None, para['batch_size'],
        data_dim=data_dim, labels2mat=labels2mat, channel=len(para['classes']),
        project_name=para['project'], dev=para['device'])
    train_loader, train_num_tot = tne.pre_process_dataset(
        train_loader, para, para['device'])

    imgs_var = tc.rand((para_g['num_samples'], 1) +
                       (para['img_size'], round(para['img_size'][1]/2)),
                       device=para_g['device'], dtype=tc.float32)
    imgs_var = nn.Parameter({imgs_var}, requires_grad=True)

    imgs = test_loader[0][0][:para_g['num_samples'], :, 0, :, :]
    imgs[:, :, :, :round(para['img_size'][1]/2)] = imgs_var
    labels = tc.ones((para_g['num_samples'], ),
                     device=para_g['device'], dtype=tc.int64) * para_g['class']

    imgs[:, 0, :, :round(para['img_size'][1]/2)] = \
        test_loader[0][0][:para['num_samples'], 0, :, :round(para['img_size'][1]/2)]

    loss_func = eval('tc.nn.' + para['loss_func'] + '()')
    optimizer = eval('tc.optim.' + para_g['optimizer'] +
                     '({imgs}, lr=para_g[\'lr\'])')

    with tc.no_grad():
        y = tc.sigmoid(imgs)
        y = tn(y)
        loss = loss_func(y, labels)
        y1 = tne.log_fidelity_cos_sin(train_loader, tc.sigmoid(imgs))
        loss = loss - y1

    for t in range(para_g['it_time']):
        y = tc.sigmoid(imgs)
        if (t % para_g['check_time']) == 0:
            print('At t = ' + str(t) + (': loss = %g' % loss.data.item()))
            bf.show_multiple_images_v1(y, save_name=para_g['img_file_name']+str(t))
        y = tn(y)
        y = nn.LogSoftmax(dim=1)(y)
        y1 = tne.log_fidelity_cos_sin(train_loader, tc.sigmoid(imgs))
        loss = -y.sum() - y1
        # loss = nn.NLLLoss()(y + y1, labels)

        # loss = loss_func(y, labels)
        with tc.no_grad():
            train_loss = loss.data.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


