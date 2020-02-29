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
    parser.add_argument('--epochs', default=15, type=int, metavar='N',
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

    if not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)
    # make a separate folder for experiment
    run_folder = '{}/{}'.format(args.save_folder, args.run_folder)
    if not os.path.isdir(run_folder):
        os.mkdir(run_folder)
        os.mkdir(run_folder+'/data')
        os.mkdir(run_folder+'/checkpoint')
        os.mkdir(run_folder+'/descr')

    # lists for full datasets
    orig_train_list_file = '{}/{}'.format(args.save_folder, 'processed/train_list.txt')
    val_list_file = '{}/{}'.format(args.save_folder, 'processed/val_list.txt')
    if (args.imbalance_ratio==1) and (args.noisy_ratio==0.0): # use original training dataset
        full_train_list_file = orig_train_list_file
    else:
        R = 1000 # number of classes
        distorted_train_list_file = '{}/{}/full_train_list_ir_{}_nr_{}.txt'.format(args.save_folder, args.run_folder, args.imbalance_ratio, args.noisy_ratio)
        full_train_list_file = distorted_train_list_file
        if not os.path.isfile(distorted_train_list_file):
            with open(orig_train_list_file) as f:
                lines = f.readlines()
            full_train_list = [x.strip().split() for x in lines]
            R = 1000 # number of classes
            class_index = random.sample(range(R), R>>1) # randomly sample half of classes which we will modify
            # class imbalance
            if args.imbalance_ratio != 1:
                distorted_list = list()
                for c in range(R):
                    c_list = [x for i, x in enumerate(full_train_list) if int(x[1]) == c]
                    A = len(c_list)
                    # select indices we will evict from the list to distort dataset
                    selected_index = list()
                    if c in class_index:
                        selected_index = random.sample(range(A), round(A*(args.imbalance_ratio-1)/args.imbalance_ratio))
                    #
                    distorted_list.extend([i for j, i in enumerate(c_list) if j not in selected_index])
                    print(c, A, len(selected_index), len(distorted_list))
            else:
                distorted_list = full_train_list
            #
            print('Imbalance =', len(distorted_list), 'selected from original', len(full_train_list))
            # noisy labels
            if args.noisy_ratio != 0.0:
                P = len(distorted_list)
                K = int(P * args.noisy_ratio)
                print('Noisy =', K, ' out of', P)
                noisy_index = random.sample(range(P), K)
                for j, i in enumerate(distorted_list): # SHOULD BE SLOW!!!
                    if j in noisy_index:
                        distorted_list[j][1] = random.randint(0,R-1)
            #
            with open(distorted_train_list_file, "w") as f:
                for item in distorted_list:
                    f.write("%s %s\n" % (item[0], item[1]))
    

    # initially we use unsupervised pretraining
    unsup_prefix = 'unsup_'
    refer_prefix = ''
    unsup_postfix = '{}batch_0_ir_{}_nr_{}_sub_{}_aug_{}'.format(unsup_prefix, args.imbalance_ratio, args.noisy_ratio, 'none', 'none')
    refer_postfix = '{}batch_0_ir_{}_nr_{}_sub_{}_aug_{}'.format(refer_prefix, args.imbalance_ratio, args.noisy_ratio, 'none', 'none')
    train_list_file = '{}/{}/train_list_{}.txt'.format(args.save_folder, args.run_folder, unsup_postfix)
    index_list_file = '{}/{}/index_list_{}.npy'.format(args.save_folder, args.run_folder, unsup_postfix)
    if os.path.isfile(train_list_file) and os.path.isfile(index_list_file):
        print('Train list exists =', train_list_file)
        with open(train_list_file) as f:
            train_list = f.readlines()
    else:
        with open(full_train_list_file) as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        index_list = range(len(lines))
        train_list = lines
        #
        np.save(index_list_file, index_list)
        with open(train_list_file, "w") as f:
            f.write("\n".join(train_list))
        print('Train list files created =', index_list_file, train_list_file)

    

    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=train_dir, file_list=train_list_file, crop=crop_size, local_rank=args.local_rank, world_size=args.world_size, dali_cpu=args.dali_cpu)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=val_dir, file_list=val_list_file, crop=crop_size, local_rank=args.local_rank, world_size=args.world_size, size=val_size)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    model_folder = '{}/{}/checkpoint'.format(args.save_folder, args.run_folder)
    if args.ensemble_size > 1:
        checkpoint_refer_file = '{}/init_{}_E_{}.pt'.format(model_folder, refer_postfix, args.ensemble_index)
        checkpoint_unsup_file = '{}/init_{}_E_{}.pt'.format(model_folder, unsup_postfix, args.ensemble_index)
    else:
        checkpoint_refer_file = '{}/init_{}.pt'.format(model_folder, refer_postfix)
        checkpoint_unsup_file = '{}/init_{}.pt'.format(model_folder, unsup_postfix)
    # save reference checkpoint (randomly initialized)
    if os.path.isfile(checkpoint_refer_file):
        print('Model {} is already trained!'.format(checkpoint_refer_file))
    else:
        print("=> creating reference model '{}'".format(args.arch))
        modelRefer = models.__dict__[args.arch](UNSUP=False)
        modelRefer = modelRefer.cuda()
        if args.fp16:
            modelRefer = network_to_half(modelRefer)
        if args.distributed:
            # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
            # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
            modelRefer = DDP(modelRefer, delay_allreduce=True)
        # evaluate on validation set
        criterion = nn.CrossEntropyLoss().cuda()
        [refer_prec1, refer_prec5] = validate(args, val_loader, modelRefer, criterion, unsup=False)
        val_loader.reset()
        #
        print('Saving reference checkpoint at epoch {} with accuracy {}'.format(0, refer_prec1))
        save_checkpoint({
            'epoch': 0,
            'arch': args.arch,
            'state_dict': modelRefer.state_dict(),
            'acc': refer_prec1,
        }, checkpoint_refer_file)
        del modelRefer, criterion
    # train unsupervised model
    if os.path.isfile(checkpoint_unsup_file):
        print('Model {} is already trained!'.format(checkpoint_unsup_file))
    else:
        print("=> creating unsupervised model '{}'".format(args.arch))
        modelUnsup = models.__dict__[args.arch](UNSUP=True)
        modelUnsup = modelUnsup.cuda()
        if args.fp16:
            modelUnsup = network_to_half(modelUnsup)
        if args.distributed:
            # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
            # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
            modelUnsup = DDP(modelUnsup, delay_allreduce=True)

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(modelUnsup.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.fp16:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss_scale, dynamic_loss_scale=args.dynamic_loss_scale)
        
        # evaluate on validation set
        [best_prec1, best_prec5] = validate(args, val_loader, modelUnsup, criterion, unsup=True)
        val_loader.reset()
        for epoch in range(args.start_epoch, args.epochs):
            # train for one epoch
            train(args, train_loader, modelUnsup, criterion, optimizer, epoch, unsup=True)
            # evaluate on validation set
            [prec1, prec5] = validate(args, val_loader, modelUnsup, criterion, unsup=True)

            # remember best prec@1 and save checkpoint
            if args.local_rank == 0:
                if prec1 > best_prec1:
                    best_prec1 = prec1
                    print('Saving best unsupervised checkpoint at epoch {} with accuracy {}'.format(epoch + 1, best_prec1))
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': modelUnsup.state_dict(),
                        'acc': best_prec1,
                        #'optimizer': optimizer.state_dict(),
                    }, checkpoint_unsup_file)
            else:
                print('Local rank is not zero')
            # reset DALI iterators
            train_loader.reset()
            val_loader.reset()
            if args.epochs == args.start_epoch - 1:
                print('##Top-1 {0}\n'
                    '##Top-5 {1}').format(prec1, prec5)


if __name__ == '__main__':
    main()
