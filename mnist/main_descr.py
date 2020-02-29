from __future__ import print_function
import argparse, os, sys, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from custom_datasets.mnist import transform_usual, MNIST
from custom_models.model import *
from utils import *

V = 10000 # number of validation images

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Descriptor-based Augmentation for PyTorch MNIST Example')
    parser.add_argument('-ts', '--train-size', type=int, default=0, metavar='N',
                        help='number of examples for training (default: 0)')
    parser.add_argument('--batch-size', type=int, default=25, metavar='N',
                        help='input batch size for training (default: 25)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--lr-decay', type=float, default=0.1, metavar='LR',
                        help='learning rate decay (default: 0.1)')
    parser.add_argument('--lr-decay_epoch', type=int, default=15, metavar='LR',
                        help='learning rate decay epoch (default: 15)')
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
    parser.add_argument('--save-folder', default='../local_data/MNIST', type=str,
                        help='dir to save data')
    parser.add_argument('-r', '--run-folder', default='run99', type=str,
                        help='dir to save run')
    parser.add_argument('-b', '--batch', type=int, default=0, metavar='N',
                        help='augmentation batch (iteration) (default: 0)')
    parser.add_argument('-as', '--augment-size', type=int, default=125, metavar='N',
                        help='augmentation dataset size for training (default: 125)')
    parser.add_argument('-sub', '--subtype-method', type=str, default='grad', metavar='N',
                        help='method to generate gradient information (default: grad)')
    parser.add_argument('-aug', '--augment-method', type=str, default='random', metavar='N',
                        help='method to match distributions (default: random)')
    parser.add_argument('-dl', '--descriptor-length', type=int, default=0, metavar='L',
                        help='descriptor length (default: 0)')
    parser.add_argument('-unsup', '--unsupervised', type=int, default=0,
                        help='unsupervised pretraining as initial step or random weights')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    init_seeds(seed=int(time.time()))

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    val_loader = torch.utils.data.DataLoader(
        MNIST(args.save_folder, args.run_folder, val=True,   transform=transform_usual),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        MNIST(args.save_folder, args.run_folder,             transform=transform_usual),
            batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = lenet(L=args.descriptor_length).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # load checkpoint
    model_folder = '{}/{}/checkpoint'.format(args.save_folder, args.run_folder)
    descr_folder = '{}/{}/descr'.format(args.save_folder, args.run_folder)
    assert os.path.isdir(model_folder), 'Error: no model checkpoint directory found!'
    assert os.path.isdir(descr_folder), 'Error: no descriptor directory found!'
    # load existing
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
    if args.ensemble_size > 1:
        index_use_file = '{}/{}/index_list_{}_E_{}.npy'.format(args.save_folder, args.run_folder, model_postfix, args.ensemble_size)
    else:
        index_use_file = '{}/{}/index_list_{}.npy'.format(args.save_folder, args.run_folder, model_postfix)
    #
    if (args.imbalance_ratio==1) and (args.noisy_ratio==0.0): # use original train dataset
        full_train_file = '{}/processed/train-split.pt'.format(args.save_folder)
    else:
        full_train_file = '{}/{}/data/train_ir_{}_nr_{}.pt'.format(args.save_folder, args.run_folder, args.imbalance_ratio, args.noisy_ratio)
    #
    _, full_targets = torch.load(full_train_file)
    
    index_all = range(full_targets.size(0))
    set_all = set(index_all)

    if os.path.isfile(checkpoint_file) and (os.path.isfile(index_use_file) or (args.batch == 0)):
        checkpoint = torch.load(checkpoint_file)
        if args.batch == 0 and args.unsupervised == 1:
            model.load_state_dict({k: v for k, v in checkpoint['state_dict'].items() if 'fc2' not in k}, strict=False) # copy all but last linear layer!
        else:
            model.load_state_dict(checkpoint['state_dict'])
        #
        if args.batch == 0:
            index_use = []
        else:
            index_use = np.load(index_use_file)
        #
        set_use = set(index_use)
        set_avail = set_all - set_use
        index_avail = list(set_avail)
        avail_size = len(index_avail)
        # checks:
        assert len(index_use) == args.train_size, 'Number of used examples should match'
    else:
        print('Some checkpoint or index files are missing in main_descr.py!')
        sys.exit(0)
    #
    val_prefix = 'val'
    train_prefix = 'train'
    if args.ensemble_size > 1:
        descr_train_file = '{}/{}_{}_E_{}.pt'.format(descr_folder, train_prefix, descr_postfix, args.ensemble_index)
        print('Descriptor files =', descr_train_file)
    else:
        descr_train_file = '{}/{}_{}.pt'.format(descr_folder, train_prefix, descr_postfix)
        descr_val_file  = '{}/{}_{}.pt'.format(descr_folder, val_prefix, descr_postfix)
        print('Descriptor files =', descr_train_file, descr_val_file)
    # augmentaion methods
    if args.augment_method == 'random':
        print('No descriptors needed in random augmentation case!')
        topindex_sim = random.sample(range(avail_size), args.augment_size)
    elif ('topUncert' in args.augment_method) and (args.ensemble_size == 1) and os.path.isfile(descr_train_file):
        print('Uncertainty-based without ensembling!')
        if args.batch == 0: # for some reason random sampling is better initially (same setup as paper epxeriments)
            print('!!!!!!!!!!!!!!!!!!!!! Random iteration when b=0 !!!!!!!!!!!!!!!!!!!!!')
            topindex_sim = random.sample(range(avail_size), args.augment_size)
        else:
            descr_train = torch.load(descr_train_file)
            assert descr_train.size(0) == full_targets.size(0), 'Number of train descriptors should be equal to number of entries in full train file'
            descr_avail = descr_train[index_avail]           
            print(descr_avail.size(), args.augment_size)
            _, topkindex_sim = torch.topk(descr_avail, args.augment_size, largest=True)
            topindex_sim = topkindex_sim.tolist()
    elif ('topUncert' in args.augment_method) and (args.ensemble_size > 1) and os.path.isfile(descr_train_file):
        print('Uncertainty-based with ensembling!')
        ensemble_index_file = '{}/{}/ensemble_list_{}_E_{}.npy'.format(args.save_folder, args.run_folder, model_postfix, args.ensemble_size)
        if args.batch == 0: # for some reason random sampling is better initially (same setup as paper epxeriments)
            print('!!!!!!!!!!!!!!!!!!!!! Random iteration when b=0 !!!!!!!!!!!!!!!!!!!!!')
            topindex_sim = random.sample(range(avail_size), args.augment_size)
        else:
            if args.ensemble_index == 0: # generate augmentation list only once
                descr_train = torch.load(descr_train_file)
                assert descr_train.size(0) == full_targets.size(0), 'Number of train descriptors should be equal to number of entries in full train file'
                descr_avail = descr_train[index_avail].to(device) # PxCx2
                for e in range(1, args.ensemble_size): # average ensemble results
                    descr_train_file = '{}/{}_{}_E_{}.pt'.format(descr_folder, train_prefix, descr_postfix, e)
                    if os.path.isfile(descr_train_file):
                        descr_train = torch.load(descr_train_file)
                        descr_avail = descr_avail + descr_train[index_avail].to(device)
                    else:
                        print('Some descriptor files are missing in ensemble-based methods')
                        sys.exit(0) 
                #
                P = descr_avail.size(0)
                r = 1e-8 # regularization for log() to avoid nan
                if 'ent' in args.subtype_method:
                    pT = descr_avail[:,:,0] # PxC
                    max_entropy = -torch.sum(torch.mul(pT, torch.log2(pT+r)), 1) # P
                    f = max_entropy
                elif 'bald' in args.subtype_method:
                    pT = descr_avail[:,:,0] # PxC
                    pL = descr_avail[:,:,1] # PxC
                    max_entropy = -torch.sum(torch.mul(pT, torch.log2(pT+r)), 1) # P
                    bald = max_entropy + torch.sum(pL, 1) # P
                    f = bald
                elif 'var' in args.subtype_method:
                    pT = descr_avail[:,:,0] # PxC
                    fMax, _ = torch.max(pT, 1)
                    var_ratio = 1 - fMax/args.ensemble_size
                    f = var_ratio
                else:
                    sys.exit(0)
                    print('Wrong ensemble uncert method!')
                #
                fCpu = f.cpu()
                _, topkindex_sim = torch.topk(fCpu, args.augment_size, largest=True)
                topindex_sim = topkindex_sim.tolist()
                np.save(ensemble_index_file, topindex_sim)
            else: # reuse existing ensemble list
                topindex_sim = np.load(ensemble_index_file)
    elif ('topK' in args.augment_method) and os.path.isfile(descr_train_file) and os.path.isfile(descr_val_file):
        descr_train = torch.load(descr_train_file)
        assert descr_train.size(0) == full_targets.size(0), 'Number of train descriptors should be equal to number of entries in full train file'
        descr_val = torch.load(descr_val_file)
        assert descr_val.size(0) == V, 'Number of val descriptors should be equal to 10000'
        index_miss = get_miss(args, model, device, val_loader)
        descr_miss = descr_val[index_miss].to(device)
        descr_avail = descr_train[index_avail].to(device)
        # some constants to calculate sub similarity matrices
        L = args.descriptor_length
        M = descr_miss.size(0)
        P = descr_avail.size(0)
        K = 32 # calculate extra because of potential overlap
        S = 2048 # divide MxL matrix into SxL chunks to fit into memory
        print('Feature-matching augmentation with descriptors:', len(index_avail), descr_val.size(), descr_miss.size(), descr_train.size(), descr_avail.size())
        # make a list for multiscale aggregation
        if   L == 90:
            I = [0, 10, 30, 80, 90]
        elif L == 80:
            I = [0, 10, 30, 80]
        elif L == 30:
            I = [0, 10, 30]
        elif L == 20:
            I = [0, 20]
        elif L == 10:
            I = [0, 10]
        elif L == 50:
            I = [0, 50]
        elif L == 0:
            I = []
        else:
            print('Wrong descriptor length {} in main_descr.py'.format(L))
            sys.exit(0)
        #
        fvec_miss  = descr_miss[ :,:,0] # MxL
        grad_miss  = descr_miss[ :,:,1] # MxL
        fvec_avail = descr_avail[:,:,0] # NxL
        grad_avail = descr_avail[:,:,1] # NxL
        #
        eps = 1e-10 # small regularization constant
        # PCC stuff
        if 'Pcc' in args.augment_method:
            for i in range(1, len(I)):
                fvec_miss[ :, I[i-1]:I[i]] -= torch.mean(fvec_miss[ :, I[i-1]:I[i]], 1, keepdim=True)
                fvec_avail[:, I[i-1]:I[i]] -= torch.mean(fvec_avail[:, I[i-1]:I[i]], 1, keepdim=True)
                grad_miss[ :, I[i-1]:I[i]] -= torch.mean(grad_miss[ :, I[i-1]:I[i]], 1, keepdim=True)
                grad_avail[:, I[i-1]:I[i]] -= torch.mean(grad_avail[:, I[i-1]:I[i]], 1, keepdim=True)
                fvec_miss[ :, I[i-1]:I[i]] /= (torch.std(fvec_miss[ :, I[i-1]:I[i]], 1, keepdim=True) + eps)
                fvec_avail[:, I[i-1]:I[i]] /= (torch.std(fvec_avail[:, I[i-1]:I[i]], 1, keepdim=True) + eps)
                grad_miss[ :, I[i-1]:I[i]] /= (torch.std(grad_miss[ :, I[i-1]:I[i]], 1, keepdim=True) + eps)
                grad_avail[:, I[i-1]:I[i]] /= (torch.std(grad_avail[:, I[i-1]:I[i]], 1, keepdim=True) + eps)
        #
        if args.augment_size >= M:
            kidx = range(M)
        else:
            print('K-center clustering for K/M:', args.augment_size, M)
            df = fvec_miss
            dg = grad_miss
            # copy
            dfA = df
            dgA = dg
            # k-centers:
            kidx = list()
            kidx.append(random.sample(range(M), 1)[0]) # initial center
            for b in range(1, args.augment_size):
                K = len(kidx)
                dfB = torch.index_select(df, 0, torch.tensor(kidx, dtype=torch.long).to(device)) # df[kidx] # KxL
                fDis = torch.mm(dfB, dfA.t()) # KxL * LxM = KxM
                dis = fDis
                if ('Grad' in args.augment_method):# and (args.batch > 0):
                    dgB = torch.index_select(dg, 0, torch.tensor(kidx, dtype=torch.long).to(device)) # dg[kidx] # KxL
                    gDis = torch.mm(dgB, dgA.t()) # KxL * LxM = KxM
                    dis += gDis
                #
                fCand = torch.max(dis, dim=0)[0]
                iCand = torch.argmin(fCand, dim=0)
                kidx.append(iCand.item())
        #
        fvec_k_miss = fvec_miss[kidx].clone()
        grad_k_miss = grad_miss[kidx].clone()
        #
        fA = fvec_avail
        gA = grad_avail
        #
        KM = len(kidx)
        C = KM // S # number of chunks
        #
        topkindex_sim = torch.zeros((K, KM), dtype=torch.long)
        topkvalue_sim = torch.zeros((K, KM)                  )
        print('fA/fB', fA.size(), fvec_k_miss.size())
        #
        for c in range(C+1):
            idx = range(c*S, min((c+1)*S, KM))
            fB = fvec_k_miss[idx]
            fSim = torch.mm(fA, fB.t()) # PxL * LxS = PxS
            sim = fSim
            if ('Grad' in args.augment_method):# and (args.batch > 0):
                gB = grad_k_miss[idx]
                gSim = torch.mm(gA, gB.t()) # PxL * LxS = PxS
                sim += gSim
            #
            simCpu = sim.cpu()
            sim_val, sim_idx = torch.topk(simCpu, K, dim=0, largest=True) # KxS
            topkindex_sim[:, idx] = sim_idx
            topkvalue_sim[:, idx] = sim_val
        #
        del descr_miss, descr_avail
        # topkindex_sim KxKM
        sortVal, sortIdx = torch.sort(topkvalue_sim, dim=1, descending=True) # KxKM
        sortedindex_sim = torch.zeros((K, KM), dtype=torch.long)
        for k in range(K):
            sortedindex_sim[k] = topkindex_sim[k, sortIdx[k]]
        sortedindex_sim = sortedindex_sim.view(-1)
        topindex_sim = sortedindex_sim[0:args.augment_size]
        i = 1
        while torch.unique(topindex_sim).size(0) != args.augment_size:
            topindex_sim = torch.cat([topindex_sim, sortedindex_sim[args.augment_size+i].view(1)])
            i = i + 1
        print('Search Iteration =', i, topindex_sim.size(0), torch.unique(topindex_sim).size(0))
        topindex_sim = torch.unique(topindex_sim)
        topindex_sim = topindex_sim.tolist()
    else:
        print('Wrong augmentation method or some descriptor files are missing')
        sys.exit()

    index_sim = [index_avail[i] for i in topindex_sim]
    set_sim = set(index_sim)
    augment_index_list = list(set_use | set_sim)
    assert len(augment_index_list) == args.train_size + args.augment_size, ' Augmented train list length is wrong: {} vs. {} + {}'.format(len(augment_index_list), args.train_size, args.augment_size)
    # update train list
    augment_postfix = '{}batch_{}_size_{}_ir_{}_nr_{}_sub_{}_aug_{}_L_{}'.format(unsup_prefix, args.batch+1,
            args.train_size + args.augment_size, args.imbalance_ratio, args.noisy_ratio, args.subtype_method, args.augment_method, args.descriptor_length)
    #
    if args.ensemble_size > 1:
        augment_checkpoint_file = '{}/best_{}_E_{}.pt'.format(model_folder, augment_postfix, args.ensemble_index)
        augment_index_list_file = '{}/{}/index_list_{}_E_{}.npy'.format(args.save_folder, args.run_folder, augment_postfix, args.ensemble_size)
    else:
        augment_checkpoint_file = '{}/best_{}.pt'.format(model_folder, augment_postfix)
        augment_index_list_file = '{}/{}/index_list_{}.npy'.format(args.save_folder, args.run_folder, augment_postfix)
    #
    np.save(augment_index_list_file, augment_index_list)

    train_loader = torch.utils.data.DataLoader(
        MNIST(args.save_folder, args.run_folder, train=True, transform=transform_usual, train_list=augment_index_list,
            imbalance_ratio=args.imbalance_ratio, noisy_ratio=args.noisy_ratio),
            batch_size=args.batch_size, shuffle=True, **kwargs)

    #print('######## THIS IS AUGMENTED MNIST NETWORK RUN ########')
    acc = test(args, model, device, test_loader, 0)
    save(model, acc, 0, augment_checkpoint_file)
    best_acc = acc
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        acc = test(args, model, device, test_loader, epoch)
        # save checkpoint
        if args.save_model and acc > best_acc:
            print('Saving...', epoch)
            save(model, acc, epoch, augment_checkpoint_file)
            best_acc = acc


if __name__ == '__main__':
    main()
