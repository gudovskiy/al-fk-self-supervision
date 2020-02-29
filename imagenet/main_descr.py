import argparse, os, random, shutil, time, warnings, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from custom_datasets.dali import HybridTrainPipe, HybridValPipe
import custom_models as models
from custom_models.model import *

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

V = 50000

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='*',
                        help='path(s) to dataset (if one path is provided, it is assumed\n' +
                        'to have subdirectories named "train" and "val"; alternatively,\n' +
                        'train and val paths can be specified directly by providing both paths as arguments)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=60, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-bs', '--batch-size', default=128, type=int, metavar='N',
                        help='batch size for descriptor generation (default: 128)')
    parser.add_argument('-lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                        help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    #parser.add_argument('--evaluate', dest='evaluate', action='store_true',
    #                    help='evaluate model on validation set')
    parser.add_argument('--fp16', action='store_true',
                        help='Run model fp16 mode.')
    parser.add_argument('--dali_cpu', action='store_true',
                        help='Runs CPU based version of DALI pipeline.')
    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument('--dynamic-loss-scale', action='store_true',
                        help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                        '--static-loss-scale.')
    parser.add_argument('--prof', dest='prof', action='store_true',
                        help='Only run 10 iterations for profiling.')
    parser.add_argument('-t', '--test', action='store_true',
                        help='Launch test mode with preset arguments')

    parser.add_argument("--local_rank", default=0, type=int)
    # added
    parser.add_argument('-ts', '--train-size', type=int, default=0, metavar='N',
                        help='number of examples for training (default: 0)')
    parser.add_argument('-ir', '--imbalance-ratio', type=int, default=1, metavar='N',
                        help='ratio of 0..499 to 500..999 labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-nr', '--noisy-ratio', type=float, default=0.0, metavar='N',
                        help='ratio of noisy(random) labels in the training dataset drawn from uniform distribution')
    parser.add_argument('-ens', '--ensemble-size', type=int, default=1, metavar='E',
                        help='defines size of ensemble or, by default, no ensemble if = 1')
    parser.add_argument('-e', '--ensemble-index', type=int, default=0, metavar='E',
                        help='defines index of ensemble')
    parser.add_argument('--save-folder', default='../local_data/ImageNet', type=str,
                        help='dir to save data')
    parser.add_argument('-r', '--run-folder', default='run99', type=str,
                        help='dir to save run')
    parser.add_argument('-b', '--batch', type=int, default=0, metavar='N',
                        help='augmentation batch (iteration) (default: 0)')
    parser.add_argument('-as', '--augment-size', type=int, default=64000, metavar='N',
                        help='augmentation dataset size for training (default: 64000)')
    parser.add_argument('-sub', '--subtype-method', type=str, default='grad', metavar='N',
                        help='method to generate gradient information (default: grad)')
    parser.add_argument('-aug', '--augment-method', type=str, default='random', metavar='N',
                        help='method to match distributions (default: random)')
    parser.add_argument('-dl', '--descriptor-length', type=int, default=0, metavar='L',
                        help='descriptor length (default: 0)')
    parser.add_argument('-unsup', '--unsupervised', type=int, default=0,
                        help='unsupervised pretraining as initial step or random weights')

    args = parser.parse_args()
    cudnn.benchmark = True

    # test mode, use default args for sanity test
    if args.test:
        args.fp16 = False
        args.epochs = 1
        args.start_epoch = 0
        args.arch = 'resnet18'
        args.batch_size = 256
        args.data = []
        args.prof = True
        args.data.append('/data/imagenet/train-jpeg/')
        args.data.append('/data/imagenet/val-jpeg/')

    if not len(args.data):
        raise Exception("error: too few data arguments")

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

    # Data loading code
    if len(args.data) == 1:
        train_dir = os.path.join(args.data[0], 'train')
        val_dir = os.path.join(args.data[0], 'val')
    else:
        train_dir = args.data[0]
        val_dir= args.data[1]

    if(args.arch == "inception_v3"):
        crop_size = 299
        val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256
    #
    print('Running main_descr.py with {} {} {} {} {}'.format(args.batch, args.train_size, args.augment_size, args.subtype_method, args.augment_method))
    # pipe for val dataset
    val_list_file = '{}/{}'.format(args.save_folder, 'processed/val_list.txt')
    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=val_dir,   file_list=val_list_file,   crop=crop_size, local_rank=args.local_rank, world_size=args.world_size, size=val_size)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](L=args.descriptor_length)
    model = model.cuda()
    if args.fp16:
        model = network_to_half(model)
    if args.distributed:
        # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
        # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss_scale, dynamic_loss_scale=args.dynamic_loss_scale)

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
        full_train_list_file = '{}/processed/train_list.txt'.format(args.save_folder)
    else:
        full_train_list_file = '{}/{}/full_train_list_ir_{}_nr_{}.txt'.format(args.save_folder, args.run_folder, args.imbalance_ratio, args.noisy_ratio)
    #
    with open(full_train_list_file) as f:
        full_train_list = f.readlines()
    full_train_list = [l.strip() for l in full_train_list]
    
    index_all = range(len(full_train_list))
    set_all = set(index_all)

    if os.path.isfile(checkpoint_file) and (os.path.isfile(index_use_file) or (args.batch == 0)):
        checkpoint = torch.load(checkpoint_file)
        if args.batch == 0 and args.unsupervised == 1:
            model.load_state_dict({k: v for k, v in checkpoint['state_dict'].items() if 'fc' not in k}, strict=False) # copy all but last linear layer!
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
            assert descr_train.size(0) == len(full_train_list), 'Number of train descriptors should be equal to number of entries in full train file'
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
                assert descr_train.size(0) == len(full_train_list), 'Number of train descriptors should be equal to number of entries in full train file'
                descr_avail = descr_train[index_avail].cuda() # PxCx2
                for e in range(1, args.ensemble_size): # average ensemble results
                    descr_train_file = '{}/{}_{}_E_{}.pt'.format(descr_folder, train_prefix, descr_postfix, e)
                    if os.path.isfile(descr_train_file):
                        descr_train = torch.load(descr_train_file)
                        descr_avail = descr_avail + descr_train[index_avail].cuda()
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
        assert descr_train.size(0) == len(full_train_list), 'Number of train descriptors should be equal to number of entries in full train file'
        descr_val = torch.load(descr_val_file)
        assert descr_val.size(0) == V, 'Number of val descriptors should be equal to 50000'
        index_miss = get_miss(args, val_loader, model, criterion)
        index_miss = [v for v in index_miss if v < V] # workaround for DALI bug
        val_loader.reset()
        descr_miss = descr_val[index_miss].cuda()
        descr_avail = descr_train[index_avail].cuda()
        # some constants to calculate sub similarity matrices
        L = args.descriptor_length
        M = descr_miss.size(0)
        P = descr_avail.size(0)
        K = 128 # calculate extra because of potential overlap
        S = 128 # divide MxL matrix into SxL chunks to fit into memory
        print('Feature-matching augmentation with descriptors:', len(index_avail), descr_val.size(), descr_miss.size(), descr_train.size(), descr_avail.size())
        # make a list for multiscale attention
        if   L == 448:
            I = [0, 64, 192, 448]
        elif L == 512:
            I = [0, 512]
        elif L == 768:
            I = [0, 256, 768]
        elif L == 0:
            I = []
        else:
            print('Wrong descriptor length in main_descr.py')
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
                dfB = torch.index_select(df, 0, torch.tensor(kidx, dtype=torch.long).cuda()) # df[kidx] # KxL
                fDis = torch.mm(dfB, dfA.t()) # KxL * LxM = KxM
                dis = fDis
                if ('Grad' in args.augment_method):# and (args.batch > 0):
                    dgB = torch.index_select(dg, 0, torch.tensor(kidx, dtype=torch.long).cuda()) # dg[kidx] # KxL
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
        augment_train_list_file = '{}/{}/train_list_{}_E_{}.txt'.format(args.save_folder, args.run_folder, augment_postfix, args.ensemble_size)
        augment_index_list_file = '{}/{}/index_list_{}_E_{}.npy'.format(args.save_folder, args.run_folder, augment_postfix, args.ensemble_size)
    else:
        augment_checkpoint_file = '{}/best_{}.pt'.format(model_folder, augment_postfix)
        augment_train_list_file = '{}/{}/train_list_{}.txt'.format(args.save_folder, args.run_folder, augment_postfix)
        augment_index_list_file = '{}/{}/index_list_{}.npy'.format(args.save_folder, args.run_folder, augment_postfix)
    #
    np.save(augment_index_list_file, augment_index_list)
    augment_train_list = [full_train_list[i] for i in augment_index_list]
    with open(augment_train_list_file, "w") as f:
        f.write("\n".join(augment_train_list))

    # pipe for train dataset
    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=train_dir, file_list=augment_train_list_file, crop=crop_size, local_rank=args.local_rank, world_size=args.world_size, dali_cpu=args.dali_cpu)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    [prec1, prec5] = validate(args, val_loader, model, criterion)
    save_checkpoint({
        'epoch': 0,
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'acc': prec1,
    }, augment_checkpoint_file)
    best_prec1 = prec1
    val_loader.reset()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(args, train_loader, model, criterion, optimizer, epoch)
        # evaluate on validation set
        [prec1, prec5] = validate(args, val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        if args.local_rank == 0:
            if prec1 > best_prec1:
                best_prec1 = prec1
                print('Saving best checkpoint at epoch {} with accuracy {}'.format(epoch + 1, best_prec1))
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'acc': best_prec1,
                }, augment_checkpoint_file)
        else:
            print('Local rank is not zero')

        # reset DALI iterators
        train_loader.reset()
        val_loader.reset()

        print('##Top-1 {}, Top-5 {}'.format(prec1, prec5))


if __name__ == '__main__':
    main()
