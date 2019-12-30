import matplotlib.pyplot as plt
import numpy as np
import torch as tc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def show_multiple_images_v1(imgs, lxy=None, titles=None, save_name=None, cmap=None):
    if cmap is None:
        cmap = plt.cm.gray
    ni = len(imgs)
    if lxy is None:
        lx = int(np.sqrt(ni)) + 1
        ly = int(ni / lx) + 1
    else:
        lx, ly = tuple(lxy)
    plt.figure()
    for n in range(ni):
        plt.subplot(lx, ly, n + 1)
        tmp = imgs[n].cpu().numpy()
        if tmp.ndim == 2:
            plt.imshow(tmp, cmap=cmap)
        else:
            plt.imshow(tmp)
        if titles is not None:
            plt.title(str(titles[n]))
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
    if type(save_name) is str:
        plt.savefig(save_name)
    plt.show()


def scatter3d(x, y, z, if_plot=True):
    if if_plot:
        plt.close()
    if type(x) is tc.Tensor:
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        z = z.cpu().numpy()
    ax1 = plt.figure().add_subplot(111, projection='3d')
    ax1.set_title('Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    ax1.scatter(x, y, z, c='r', marker='o')
    plt.legend('x1')
    ax1.view_init(elev=90, azim=0)
    if if_plot:
        plt.draw()
        plt.pause(1)
    return ax1


def surf(x=None, y=None, z=None, xlabel='x label', ylabel='y label', zlabel='z label', title=''):
    fig = plt.figure()
    ax = Axes3D(fig)
    if x is None:
        x = np.array(range(z.shape[1]))
    elif type(x) is tc.Tensor:
        x = x.cpu().numpy()
    if y is None:
        y = np.array(range(z.shape[0]))
    elif type(y) is tc.Tensor:
        y = y.cpu().numpy()
    if type(z) is tc.Tensor:
        z = z.cpu().numpy()
    x, y = np.meshgrid(x, y)
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet,
                           linewidth=0, antialiased=False)
    ax.set_xlabel(xlabel, color='r')
    ax.set_ylabel(ylabel, color='r')
    ax.set_zlabel(zlabel)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.suptitle(title)
    plt.show()


def plot(x, *y, marker='s'):
    if type(x) is tc.Tensor:
        if x.device != 'cpu':
            x = x.cpu()
        x = x.numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if len(y) > 0.5:
        for y0 in y:
            if type(y0) is tc.Tensor:
                if y0.device != 'cpu':
                    y0 = y0.cpu()
                y0 = y0.numpy()
            ax.plot(x, y0, marker=marker)
    else:
        ax.plot(x, marker=marker)
    plt.show()


def plot_v1(x, *y, options=None, save=None):
    if options is None:
        options = dict()
    num_curves = max(1, y.__len__())
    # Default font
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 40}
    # Default values
    label_names = list()
    for n in range(num_curves):
        label_names.append('curve-' + str(n))
    default_ops = ['labelsize', 'axfontname', 'labelfont', 'axnames',
                   'legendfont', 'markers', 'labelnames']
    default_val = [32, 'Times New Roman', font1, ['', ''], font1,
                   ['o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+'],
                   label_names]

    save_opts = dict()
    save_opts['name'] = 'img.png'
    if type(save) is dict:
        for s in save:
            save_opts[s] = save[s]

    opts = dict()
    for n in range(default_ops.__len__()):
        if default_ops[n] in options:
            opts[default_ops[n]] = options[default_ops[n]]
        else:
            opts[default_ops[n]] = default_val[n]
    while num_curves > opts['markers'].__len__():
        opts['markers'] = opts['markers'] * 2
    opts['markers'] = opts['markers'][:num_curves]

    # start plotting
    if type(x) is tc.Tensor:
        if x.device != 'cpu':
            x = x.cpu()
        x = x.numpy()
    x = np.array(x).reshape(-1,)
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(1, 1, 1)
    if len(y) > 0.5:
        n = 0
        for y0 in y:
            if type(y0) is tc.Tensor:
                if y0.device != 'cpu':
                    y0 = y0.cpu()
                y0 = y0.numpy()
            ax.plot(x, np.array(y0).reshape(-1,), marker=opts['markers'][n], markerfacecolor='w', markersize=12,
                    markeredgewidth=2, label=opts['labelnames'][n])
            n += 1
    else:
        ax.plot(x, marker=opts['markers'][0], markerfacecolor='w', markersize=12,
                markeredgewidth=2, label=opts['labelnames'][0])

    plt.tick_params(labelsize=opts['labelsize'])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(opts['axfontname']) for label in labels]

    plt.xlabel(opts['axnames'][0], opts['labelfont'])
    plt.ylabel(opts['axnames'][1], opts['labelfont'])

    plt.legend(prop=opts['legendfont'])
    if type(save) is dict:
        # plt.subplots_adjust(left=0.09, right=1, wspace=0.25, hspace=0.25, bottom=0.13, top=0.91)
        plt.savefig(save_opts['name'])
    plt.show()



