import argparse, os, random, shutil, time, warnings, sys
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
    parser = argparse.ArgumentParser(description='Descriptor Generator for PyTorch ImageNet Example')
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
    parser.add_argument('-bs', '--batch-size', default=256, type=int, metavar='N',
                        help='batch size for descriptor generation (default: 256)')
    parser.add_argument('-p', '--print-freq', default=50, type=int, metavar='N',
                        help='print frequency (default: 50)')
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
    cudnn.benchmark = True

    # test mode, use default args for sanity test
    if args.test:
        args.fp16 = False
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

    # lists for full datasets
    val_list_file = '{}/{}'.format(args.save_folder, 'processed/val_list.txt')
    if (args.imbalance_ratio==1) and (args.noisy_ratio==0.0): # use original training dataset
        train_list_file = '{}/{}'.format(args.save_folder, 'processed/train_list.txt')
    else:
        train_list_file = '{}/{}/full_train_list_ir_{}_nr_{}.txt'.format(args.save_folder, args.run_folder, args.imbalance_ratio, args.noisy_ratio)

    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=train_dir, file_list=train_list_file, crop=crop_size, local_rank=args.local_rank, world_size=args.world_size, size=val_size)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, data_dir=val_dir,   file_list=val_list_file,   crop=crop_size, local_rank=args.local_rank, world_size=args.world_size, size=val_size)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    # create model
    print("=> creating model '{} {}'".format(args.arch, args.descriptor_length))
    if 'MC' in args.subtype_method:
        model = models.__dict__[args.arch](L=args.descriptor_length, MC=True)
    else:
        model = models.__dict__[args.arch](L=args.descriptor_length, MC=False)
    #
    model = model.cuda()
    if args.fp16:
        model = network_to_half(model)
    if args.distributed:
        # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
        # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-0)
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss_scale, dynamic_loss_scale=args.dynamic_loss_scale)

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
            gen_mc(args, train_loader, model, criterion, optimizer, train_prefix, descr_train_file)
    else:
        print('Generating val descriptors')
        gen_descr(args, val_loader, model, criterion, optimizer, val_prefix, descr_val_file, descr_val_file)
        print('Generating train descriptors')
        gen_descr(args, train_loader, model, criterion, optimizer, train_prefix, descr_train_file, descr_val_file)

if __name__ == '__main__':
    main()
