from __future__ import print_function
import argparse, os, sys, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from custom_datasets.svhn import transform_usual, SVHN
from custom_models.model import *
from custom_models.resnet import *
from utils import *

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=2000, metavar='N',
                        help='input batch size for testing (default: 2000)')
    parser.add_argument('--epochs', type=int, default=8, metavar='N',
                        help='number of epochs to train (default: 8)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--lr-decay', type=float, default=0.1, metavar='LR',
                        help='learning rate decay (default: 0.1)')
    parser.add_argument('--lr-decay_epoch', type=int, default=3, metavar='LR',
                        help='learning rate decay epoch (default: 3)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='LR',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save best model when training')
    parser.add_argument('-ir', '--imbalance-ratio', type=int, default=1, metavar='N',
                        help='ratio of 0..4 to 5..9 labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-nr', '--noisy-ratio', type=float, default=0.0, metavar='N',
                        help='ratio of noisy(random) labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-ens', '--ensemble-size', type=int, default=1, metavar='E',
                        help='defines size of ensemble or, by default, no ensemble if = 1')
    parser.add_argument('-e', '--ensemble-index', type=int, default=0, metavar='E',
                        help='defines index of ensemble')
    parser.add_argument('--save-folder', default='../local_data/SVHN', type=str,
                        help='dir to save data')
    parser.add_argument('-r', '--run-folder', default='run99', type=str,
                        help='dir to save run')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    init_seeds(seed=int(time.time()))

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    imbalance_ratio = args.imbalance_ratio
    noisy_ratio = args.noisy_ratio

    if not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)
    # make a separate folder for experiment
    run_folder = '{}/{}'.format(args.save_folder, args.run_folder)
    if not os.path.isdir(run_folder):
        os.mkdir(run_folder)
        os.mkdir(run_folder+'/data')
        os.mkdir(run_folder+'/checkpoint')
        os.mkdir(run_folder+'/descr')

    test_loader = torch.utils.data.DataLoader(
        SVHN(args.save_folder, args.run_folder, download=True, transform=transform_usual),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    data_full_file = '{}/processed/train-split.pt'.format(args.save_folder)
    if os.path.isfile(data_full_file):
        data, targets = torch.load(data_full_file)
    else:
        print('Some files are missing in main.py!')
        sys.exit(0)

    if not(imbalance_ratio==1 and noisy_ratio==0.0):
        R = 10 # number of classes
        data_file = '{}/{}/data/train_ir_{}_nr_{}.pt'.format(args.save_folder, args.run_folder, imbalance_ratio, noisy_ratio)
        if os.path.isfile(data_file):
            data, targets = torch.load(data_file)
        else:
            # imbalance
            if imbalance_ratio != 1:
                maskL = targets.lt(R/2) # 0...4
                indexL = maskL.nonzero().squeeze()
                imagesL = torch.index_select(data, 0, indexL)
                labelsL = torch.index_select(targets, 0, indexL)
                L = labelsL.size(0)

                maskU = targets.ge(R/2) # 5...9
                indexU = maskU.nonzero().squeeze()
                imagesU = torch.index_select(data, 0, indexU)
                labelsU = torch.index_select(targets, 0, indexU)
                U = labelsU.size(0)

                S = int(1.0 * L / imbalance_ratio) # number of U examples
                ind = random.sample(range(U), S)

                imbalanced_images = torch.zeros((L+S, data.size(1), data.size(2), data.size(3)), dtype=torch.uint8)
                imbalanced_images[ :L  ,...] = imagesL
                imbalanced_images[L:L+S,...] = imagesU[ind]
                imbalanced_labels = torch.zeros((L+S,), dtype=torch.long)
                imbalanced_labels[ :L  ] = labelsL
                imbalanced_labels[L:L+S] = labelsU[ind]
                #
                print('Imbalance =', L, U, ':', S, '->', L+S)
                data = imbalanced_images
                targets = imbalanced_labels
            if noisy_ratio != 0.0:
                P = targets.size(0)
                K = int(P * noisy_ratio)
                print('Noisy =', K, ' out of', P)
                ind = random.sample(range(P), K)
                #
                targets[ind] = torch.randint(0, R, (K,), dtype=torch.long)
            #
            print('Distorted dataset sizes', data.size(), targets.size())
            with open(data_file, 'wb') as f:
                torch.save((data, targets), f)
    

    # initially we use unsupervised pretraining
    unsup_prefix = 'unsup_'
    refer_prefix = ''
    unsup_postfix = '{}batch_0_ir_{}_nr_{}_sub_{}_aug_{}'.format(unsup_prefix, args.imbalance_ratio, args.noisy_ratio, 'none', 'none')
    refer_postfix = '{}batch_0_ir_{}_nr_{}_sub_{}_aug_{}'.format(refer_prefix, args.imbalance_ratio, args.noisy_ratio, 'none', 'none')
    index_list_file = '{}/{}/index_list_{}.npy'.format(args.save_folder, args.run_folder, unsup_postfix)
    if os.path.isfile(index_list_file):
        print('Train list exists =', index_list_file)
        index_list = np.load(index_list_file)
    else:
        index_list = range(data.size(0))
        np.save(index_list_file, index_list)
        print('Index list created =', index_list_file)

    
    train_loader = torch.utils.data.DataLoader(
        SVHN(args.save_folder, args.run_folder, train=True, download=True, transform=transform_usual,
            train_list=index_list, imbalance_ratio=args.imbalance_ratio, noisy_ratio=args.noisy_ratio),
            batch_size=args.batch_size, shuffle=True, **kwargs)
    #
    model_folder = '{}/{}/checkpoint'.format(args.save_folder, args.run_folder)
    if args.ensemble_size > 1:
        checkpoint_refer_file = '{}/init_{}_E_{}.pt'.format(model_folder, refer_postfix, args.ensemble_index)
        checkpoint_unsup_file = '{}/init_{}_E_{}.pt'.format(model_folder, unsup_postfix, args.ensemble_index)
    else:
        checkpoint_refer_file = '{}/init_{}.pt'.format(model_folder, refer_postfix)
        checkpoint_unsup_file = '{}/init_{}.pt'.format(model_folder, unsup_postfix)
    # save reference checkpoint (randomly initialized)
    modelRefer = resnet10(UNSUP=False).to(device)
    accRefer = test(args, modelRefer, device, test_loader, 0, unsup=False)
    if not os.path.isfile(checkpoint_unsup_file):
        if args.save_model:
            print('Saving reference...')
            save(modelRefer, accRefer, 0, checkpoint_refer_file)
    del modelRefer, accRefer
    #
    if not os.path.isfile(checkpoint_unsup_file):
        modelUnsup = resnet10(UNSUP=True).to(device)
        optimizer = optim.SGD(modelUnsup.parameters(), lr=args.lr, momentum=args.momentum)
        acc = test(args, modelUnsup, device, test_loader, 0, unsup=True)
        save(modelUnsup, acc, 0, checkpoint_unsup_file)
        best_acc = acc
        for epoch in range(1, args.epochs + 1):
            train(args, modelUnsup, device, train_loader, optimizer, epoch, unsup=True)
            acc = test(args, modelUnsup, device, test_loader, epoch, unsup=True)
            # save checkpoint
            if args.save_model and acc > best_acc:
                print('Saving unsupervised...', epoch)
                save(modelUnsup, acc, epoch, checkpoint_unsup_file)
                best_acc = acc


if __name__ == '__main__':
    main()
