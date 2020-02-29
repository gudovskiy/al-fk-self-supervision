import torch
from torchvision import datasets #, transforms
import torch.utils.data.sampler  as sampler
import torch.utils.data as data

import numpy as np
import argparse
import random
import os, sys, time

from custom_datasets import *
import model
#import vgg
import resnet
from solver import Solver
from utils import init_seeds
import arguments

#sys.path.append('../mnist/custom_datasets')
#sys.path.append('../mnist/custom_models')
from mnist import mnist_transformer, MNIST
from svhn import svhn_transformer, SVHN

#def cifar_transformer():
#    return transforms.Compose([
#            transforms.RandomHorizontalFlip(),
#            transforms.ToTensor(),
#        ])

def main(args):
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    init_seeds(seed=int(time.time()))
    kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}
    print(args.dataset)

    if args.dataset == 'MNIST':
        test_dataloader = data.DataLoader(
            MNIST(args.data_path, args.run_folder, transform=mnist_transformer()),
                batch_size=10000, shuffle=False, **kwargs)

        train_dataset = MNIST(args.data_path, args.run_folder, train=True, transform=mnist_transformer(), imbalance_ratio=args.imbalance_ratio)

        if args.imbalance_ratio == 100:
            args.num_images = 25711
        else:
            args.num_images = 50000

        args.budget = 125
        args.initial_budget = 125
        args.num_classes = 10
        args.num_channels = 1
        args.arch_scaler = 2
    elif args.dataset == 'SVHN':
        test_dataloader = data.DataLoader(
            SVHN(args.data_path, args.run_folder, transform=svhn_transformer()),
                batch_size=5000, shuffle=False, **kwargs)

        train_dataset = SVHN(args.data_path, args.run_folder, train=True, transform=svhn_transformer(), imbalance_ratio=args.imbalance_ratio)

        if args.imbalance_ratio == 100:
            args.num_images = 318556
        else:
            args.num_images = 500000

        args.budget = 1250
        args.initial_budget = 1250
        args.num_classes = 10
        args.num_channels = 3
        args.arch_scaler = 1
    elif args.dataset == 'cifar10':
        test_dataloader = data.DataLoader(
                datasets.CIFAR10(args.data_path, download=True, transform=cifar_transformer(), train=False),
            batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR10(args.data_path)

        args.num_images = 50000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 10
        args.num_channels = 3
    elif args.dataset == 'cifar100':
        test_dataloader = data.DataLoader(
                datasets.CIFAR100(args.data_path, download=True, transform=cifar_transformer(), train=False),
             batch_size=args.batch_size, drop_last=False)

        train_dataset = CIFAR100(args.data_path)

        args.num_images = 50000
        args.budget = 2500
        args.initial_budget = 5000
        args.num_classes = 100
        args.num_channels = 3
    elif args.dataset == 'ImageNet':
        test_dataloader = data.DataLoader(
            ImageNet(args.data_path + '/val', transform=imagenet_test_transformer()),
                batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)

        if args.imbalance_ratio == 100:
            train_dataset = ImageNet(args.data_path + '/train_ir_100', transform=imagenet_train_transformer())
            args.num_images = 645770
        else:
            train_dataset = ImageNet(args.data_path + '/train', transform=imagenet_train_transformer())
            args.num_images = 1281167

        args.budget = 64000
        args.initial_budget = 64000
        args.num_classes = 1000
        args.num_channels = 3
        args.arch_scaler = 1
    else:
        raise NotImplementedError

    all_indices = set(np.arange(args.num_images))
    initial_indices = random.sample(all_indices, args.initial_budget)
    sampler = data.sampler.SubsetRandomSampler(initial_indices)
    #print(args.batch_size, sampler)
    # dataset with labels available
    querry_dataloader = data.DataLoader(train_dataset, sampler=sampler,
            batch_size=args.batch_size, drop_last=False, **kwargs)
    print('Sampler size =', len(querry_dataloader))
    solver = Solver(args, test_dataloader)

    splits = range(1,11)

    current_indices = list(initial_indices)

    accuracies = []
    
    for split in splits:
        print("Split =", split)
        # need to retrain all the models on the new images
        # re initialize and retrain the models
        #task_model = vgg.vgg16_bn(num_classes=args.num_classes)
        if args.dataset == 'MNIST':
            task_model = model.LeNet(num_classes=args.num_classes)
        elif args.dataset == 'SVHN':
            task_model = resnet.resnet10(num_classes=args.num_classes)
        elif args.dataset == 'ImageNet':
            task_model = resnet.resnet18(num_classes=args.num_classes)
        else:
            print('WRONG DATASET!')
        # loading pretrained
        if args.pretrained:
            print("Loading pretrained model", args.pretrained)
            checkpoint = torch.load(args.pretrained)
            task_model.load_state_dict({k: v for k, v in checkpoint['state_dict'].items() if 'fc' not in k}, strict=False) # copy all but last linear layers
        #
        vae = model.VAE(z_dim=args.latent_dim, nc=args.num_channels, s=args.arch_scaler)
        discriminator = model.Discriminator(z_dim=args.latent_dim, s=args.arch_scaler)
        #print("Sampling starts")
        unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
        unlabeled_sampler = data.sampler.SubsetRandomSampler(unlabeled_indices)
        unlabeled_dataloader = data.DataLoader(train_dataset, sampler=unlabeled_sampler,
                batch_size=args.batch_size, drop_last=False, **kwargs)
        #print("Train starts")
        # train the models on the current data
        acc, vae, discriminator = solver.train(querry_dataloader,
                                               task_model, 
                                               vae, 
                                               discriminator,
                                               unlabeled_dataloader)


        print('Final accuracy with {}% of data is: {:.2f}'.format(int(split*100.0*args.budget/args.num_images), acc))
        accuracies.append(acc)

        sampled_indices = solver.sample_for_labeling(vae, discriminator, unlabeled_dataloader)
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = data.DataLoader(train_dataset, sampler=sampler,
                batch_size=args.batch_size, drop_last=False, **kwargs)

    torch.save(accuracies, os.path.join(args.out_path, args.log_name))

if __name__ == '__main__':
    args = arguments.get_args()
    main(args)

