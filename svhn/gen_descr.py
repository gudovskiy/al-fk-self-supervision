from __future__ import print_function
import argparse, os, sys, time
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
    parser = argparse.ArgumentParser(description='Descriptor Generator for PyTorch SVHN Example')
    parser.add_argument('-ts', '--train-size', type=int, default=0, metavar='N',
                        help='number of examples for training (default: 0)')
    parser.add_argument('-gbs', '--gen-batch-size', type=int, default=2000, metavar='N',
                        help='batch size for descriptor generation (default: 2000)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
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
    parser.add_argument('-b', '--batch', type=int, default=0, metavar='N',
                        help='augmentation batch (iteration) (default: 0)')
    parser.add_argument('-sub', '--subtype-method', type=str, default='grad', metavar='N',
                        help='method to generate gradient information (default: grad)')
    parser.add_argument('-aug', '--augment-method', type=str, default='random', metavar='N',
                        help='method to match distributions (default: random)')
    parser.add_argument('-smp', '--sample-steps', type=int, default=1, metavar='N',
                        help='number of samples for estimation (default: 1)')
    parser.add_argument('-dl', '--descriptor-length', type=int, default=0, metavar='L',
                        help='descriptor length (default: 0)')
    parser.add_argument('-unsup', '--unsupervised', type=int, default=0,
                        help='unsupervised pretraining as initial step or random weights')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    init_seeds(seed=int(time.time()))

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    #
    train_loader = torch.utils.data.DataLoader(
        SVHN(args.save_folder, args.run_folder, train=True, transform=transform_usual,
            imbalance_ratio=args.imbalance_ratio, noisy_ratio=args.noisy_ratio),
            batch_size=args.gen_batch_size, shuffle=False, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        SVHN(args.save_folder, args.run_folder, val=True,   transform=transform_usual),
            batch_size=args.gen_batch_size, shuffle=False, **kwargs)
    #
    if 'MC' in args.subtype_method:
        model = resnet10(L=args.descriptor_length, MC=True).to(device)
    else:
        model = resnet10(L=args.descriptor_length, MC=False).to(device)
    #
    optimizer = optim.SGD(model.parameters(), lr=1e-0)
    # load checkpoint
    model_folder = '{}/{}/checkpoint'.format(args.save_folder, args.run_folder)
    assert os.path.isdir(model_folder), 'Error: no model checkpoint directory found!'
    #
    if args.unsupervised == 1:
        unsup_prefix = 'unsup_'
    else:
        unsup_prefix = ''
    #
    descr_postfix = '{}batch_B_ir_{}_nr_{}_sub_{}_aug_{}_L_{}'.format(unsup_prefix, # we do not save descriptors for each iteration here due to large size
        args.imbalance_ratio, args.noisy_ratio, args.subtype_method, args.augment_method, args.descriptor_length)
    if args.batch == 0:
        model_postfix = '{}batch_{}_ir_{}_nr_{}_sub_{}_aug_{}'.format(unsup_prefix, args.batch,
            args.imbalance_ratio, args.noisy_ratio, 'none', 'none')
        if args.ensemble_size > 1:
            checkpoint_file = '{}/init_{}_E_{}.pt'.format(model_folder, model_postfix, args.ensemble_index)
        else:
            checkpoint_file = '{}/init_{}.pt'.format(model_folder, model_postfix)
    else:
        model_postfix = '{}batch_{}_size_{}_ir_{}_nr_{}_sub_{}_aug_{}_L_{}'.format(unsup_prefix, args.batch,
            args.train_size, args.imbalance_ratio, args.noisy_ratio, args.subtype_method, args.augment_method, args.descriptor_length)
        if args.ensemble_size > 1:
            checkpoint_file = '{}/best_{}_E_{}.pt'.format(model_folder, model_postfix, args.ensemble_index)
        else:
            checkpoint_file = '{}/best_{}.pt'.format(model_folder, model_postfix)
    #
    print('Generating descriptors using model checkpoint:', checkpoint_file)
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        #print('TRANSFER', checkpoint['state_dict'].items())
        if args.batch == 0 and args.unsupervised == 1:
            model.load_state_dict({k: v for k, v in checkpoint['state_dict'].items() if 'fc' not in k}, strict=False) # copy all but last linear layer!
        else:
            model.load_state_dict(checkpoint['state_dict'])
    else:
        print('Some files are missing in gen_descr.py!')
        sys.exit(0)
    #
    val_prefix = 'val'
    train_prefix = 'train'
    if args.ensemble_size > 1:
        descr_val_file = '{}/{}/descr/{}_{}_E_{}.pt'.format(args.save_folder, args.run_folder, val_prefix, descr_postfix, args.ensemble_index)
        descr_train_file = '{}/{}/descr/{}_{}_E_{}.pt'.format(args.save_folder, args.run_folder, train_prefix, descr_postfix, args.ensemble_index)
    else:
        descr_val_file = '{}/{}/descr/{}_{}.pt'.format(args.save_folder, args.run_folder, val_prefix, descr_postfix)
        descr_train_file = '{}/{}/descr/{}_{}.pt'.format(args.save_folder, args.run_folder, train_prefix, descr_postfix)
    #
    if 'MC' in args.subtype_method:
        print('Generating train MC')
        with torch.no_grad():
            gen_mc(args, model, optimizer, device, train_loader, train_prefix, descr_train_file)
    else:
        print('Generating val descriptors')
        gen_descr(args, model, optimizer, device, val_loader, val_prefix, descr_val_file, descr_val_file)
        print('Generating train descriptors')
        gen_descr(args, model, optimizer, device, train_loader, train_prefix, descr_train_file, descr_val_file)

if __name__ == '__main__':
    main()
