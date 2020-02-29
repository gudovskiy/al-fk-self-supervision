from __future__ import print_function
import argparse, os, sys, random, time, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from custom_datasets.mnist import transform_usual, MNIST
from custom_models.model import *
from utils import *
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
#from tsne import bh_sne
#import fitsne

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='t-SNE Visualization for PyTorch MNIST Example')
    parser.add_argument("--gpu", default='0', type=str,
                        help='GPU device number')
    parser.add_argument('--dataset', default='MNIST', type=str,
                        help='dataset name')
    parser.add_argument('-ts', '--train-size', type=int, default=0, metavar='N',
                        help='number of examples for training (default: 0)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-ir', '--imbalance-ratio', type=int, default=1, metavar='N',
                        help='ratio of 0..4 to 5..9 labels in the training dataset drawn from uniform distribution')
    parser.add_argument('--save-folder', default='../local_data/MNIST', type=str,
                        help='dir to save data')
    parser.add_argument('-r', '--run-folder', default='run0', type=str,
                        help='dir to save run')
    parser.add_argument('-b', '--batch', type=int, default=0, metavar='N',
                        help='augmentation batch (iteration) (default: 0)')
    parser.add_argument('-dl', '--descriptor-length', type=int, default=0, metavar='L',
                        help='descriptor length (default: 0)')
    parser.add_argument('--plot-tsne', action='store_true', help='TSNE of pooled features, confusion matrix otherwise')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dataset = args.dataset
    ir = args.imbalance_ratio
    dl = args.descriptor_length
    ts = args.train_size
    plotTSNE = args.plot_tsne
    b = args.batch
    # FORMAT: [imbalance ratio, unsupervised, subtype, augmentation method, descriptor length, ensemble size]
    cfg = list()
    cfg.append([ir, 1, 'varMC',    'topUncert',          0, 1])
    cfg.append([ir, 1, 'gradFmap', 'topKandPccFprGrad', dl, 1])
    cfg.append([ir, 1, 'gradAbl',  'topKandPccFprGrad', dl, 1])

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    init_seeds(seed=int(time.time()))

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    
    test_data_file = '{}/processed'.format(args.save_folder)
    image, label = torch.load('{}/test.pt'.format(test_data_file))
    V = label.shape[0]
    image, label = image.numpy().astype('float64'), label.numpy()
    index_all = range(label.shape[0])
    image_vec = np.reshape(image, (image.shape[0], -1))
    # USE transform_test!!!
    image_vec -= np.mean(image_vec)
    image_vec /= np.std(image_vec)
    image_vec = image_vec.copy(order='C')
    print('SHAPE of' + str(image.shape) + '->' + str(image_vec.shape))
    
    test_loader = torch.utils.data.DataLoader(
        MNIST(args.save_folder, args.run_folder, transform=transform_usual),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    for c in cfg:
        model = lenet(L=dl).to(device)
        if c[1] == 1:
            unsup_prefix = 'unsup_'
        else:
            unsup_prefix = ''
        # load checkpoint
        model_folder = '{}/{}/checkpoint'.format(args.save_folder, args.run_folder)
        assert os.path.isdir(model_folder), 'Error: no model checkpoint directory found!'
        # load existing
        model_postfix = '{}batch_{}_size_{}_ir_{}_nr_0.0_sub_{}_aug_{}_L_{}'.format(unsup_prefix, b, ts, c[0], c[2], c[3], c[4])
        checkpoint_file = '{}/best_{}.pt'.format(model_folder, model_postfix)
        print(checkpoint_file)
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('Some checkpoint are missing in visualize_tsne.py!')
            sys.exit(0)
        #
        index_miss, gt, pr = get_miss_and_cm(args, model, device, test_loader)
        #index_miss = get_miss(args, model, device, test_loader)
        #index_miss, descr = get_miss_descr(args, model, device, test_loader)    
        print(str(c), index_miss.size(), gt.size(), pr.size())
        one_hot = torch.zeros(V)
        one_hot[index_miss] = 1.0
        # compute confusion matrix
        cm = confusion_matrix(gt.cpu().numpy(), pr.cpu().numpy())
        # Only use the labels that appear in the data
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        c.append(one_hot.cpu().numpy())
        c.append(cm)
        #c.append(descr.cpu().numpy())

    #image_2d = np.zeros((10000,2))
    image_2d_file = 'image_2d.npy'
    if os.path.isfile(image_2d_file):
        image_2d = np.load(image_2d_file)
    else:
        image_2d = TSNE(n_components=2, verbose=1).fit_transform(image_vec)
        #image_2d = fitsne.FItSNE(image_vec)
        np.save(image_2d_file, image_2d)
    
    print('t-SNE Check', image_2d.shape)
    # plot
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    results_file = '{}_tsne_{}'.format(dataset, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
    pdf = PdfPages(results_file+'.pdf')
    np.set_printoptions(precision=2)
    fontText = 16
    plt.rcParams.update({'font.size': 16})
    S = 3*3
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    fig.set_size_inches(np.array([17,6]), forward=True)
    # confusion matrix
    if not plotTSNE:
        print('plotting confusion matrix')
        fmt = '.2f'
        for k,c in enumerate(cfg):
            cm = cfg[k][7]
            thresh = cm.max() / 2.0
            ax[k].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax[k].text(j, i, format(cm[i,j], fmt), ha="center", va="center", color="white" if cm[i,j] > thresh else "black", fontsize = 10)
            #ax[k].text(0.25, 10.4, t[k], rotation = 0, fontsize = fontText)
            plt.tight_layout()
        ax[0].set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xlabel='Predicted label, (a)', ylabel='True label')
        ax[1].set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xlabel='Predicted label, (b)')
        ax[2].set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xlabel='Predicted label, (c)')
    # t-SNE
    else:
        print('plotting tSNE')
        t = ['(a)', '(b)', '(c)']
        for k,c in enumerate(cfg):
            s = S*cfg[k][6]+1
            im = ax[k].scatter(image_2d[:,0], image_2d[:,1], cmap=plt.get_cmap('tab10'), c=label, s=s)
            ax[k].text(-80, -75, t[k], rotation = 0, fontsize = fontText)
            plt.tight_layout()
            plt.xlim(-90, 90)
            plt.ylim(-85, 85)
        #fig.colorbar(im, ax=ax.ravel().tolist(), cmap=plt.get_cmap('tab10'), orientation='horizontal', shrink = 0.25, spacing = 'proportional', drawedges = True)
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.91, 0.1, 0.01, 0.84])
        fig.colorbar(im, cax=cbar_ax, cmap=plt.get_cmap('tab10'), orientation='vertical')
    #
    from matplotlib.patches import Ellipse
    #5
    ellipse = Ellipse(xy=(24.0, 10.0), width=38.0, height=65.0, angle=-15.0, edgecolor='k', fc='None', lw=2, ls='--')
    ax[0].add_patch(ellipse)
    ellipse = Ellipse(xy=(25.0, 25.0), width=28.0, height=28.0, angle=+0.0,  edgecolor='k', fc='None', lw=2, ls='--')
    ax[1].add_patch(ellipse)
    ellipse = Ellipse(xy=(25.0, 35.0), width=28.0, height=28.0, angle=+0.0,  edgecolor='k', fc='None', lw=2, ls='--')
    ax[2].add_patch(ellipse)
    #8
    ellipse = Ellipse(xy=(-10.0, 12.0), width=50.0, height=60.0, angle=-55.0, edgecolor='k', fc='None', lw=2, ls='--')
    ax[0].add_patch(ellipse)
    ellipse = Ellipse(xy=(-25.0,  8.0), width=35.0, height=35.0, angle=10.0,  edgecolor='k', fc='None', lw=2, ls='--')
    ax[1].add_patch(ellipse)
    ellipse = Ellipse(xy=(-25.0,  8.0), width=35.0, height=35.0, angle=10.0,  edgecolor='k', fc='None', lw=2, ls='--')
    ax[2].add_patch(ellipse)
    #9
    ellipse = Ellipse(xy=(-10.0,  -50.0), width=38.0, height=65.0, angle=+25.0, edgecolor='k', fc='None', lw=2, ls='--')
    ax[0].add_patch(ellipse)
    ellipse = Ellipse(xy=(-20.0, -30.0), width=24.0, height=24.0, angle=+0.0, edgecolor='k', fc='None', lw=2, ls='--')
    ax[1].add_patch(ellipse)
    ellipse = Ellipse(xy=(-20.0, -30.0), width=24.0, height=24.0, angle=+0.0, edgecolor='k', fc='None', lw=2, ls='--')
    ax[2].add_patch(ellipse)
    ellipse = Ellipse(xy=(  0.0, -70.0), width=22.0, height=22.0, angle=+0.0, edgecolor='k', fc='None', lw=2, ls='--')
    ax[1].add_patch(ellipse)
    ellipse = Ellipse(xy=(  0.0, -70.0), width=22.0, height=22.0, angle=+0.0, edgecolor='k', fc='None', lw=2, ls='--')
    ax[2].add_patch(ellipse)
    #
    plt.savefig(results_file+'.png', bbox_inches='tight')
    #plt.savefig(results_file+'.svg', format="svg", bbox_inches='tight')
    pdf.savefig(bbox_inches='tight')
    pdf.close()
    # gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -sOutputFile=mnist-tsne1-small.pdf mnist-tsne1.pdf
    # gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen  -sOutputFile=mnist-tsne1-small.pdf mnist-tsne1.pdf

if __name__ == '__main__':
    main()
