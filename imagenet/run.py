import argparse, os, torch
from tabulate import tabulate

parser = argparse.ArgumentParser(description='PyTorch ImageNet Run')
parser.add_argument('--dataset', default='ImageNet', type=str,
                    help='dataset name')
parser.add_argument('--imagenetdir', default='../local_data/ILSVRC2012_pytorch', type=str,
                    help='dir with ImageNet')
parser.add_argument('--datadir', default='../local_data/ImageNet', type=str,
                    help='dir to save/read data')
parser.add_argument("--gpu", default='0', type=str,
                    help='GPU device number')
parser.add_argument('--initial', action='store_true',
                    help='initial batch to generate starting models')
parser.add_argument('--run-start', type=int, default=0, metavar='R',
                    help='run start pointer (default: 0)')
parser.add_argument('--run-stop', type=int, default=1, metavar='R',
                    help='run stop pointer (default: 1)')
parser.add_argument('-smp', '--sample-steps', type=int, default=32, metavar='N',
                    help='number of samples for estimation (default: 32)')
parser.add_argument('-ens', '--ensemble-size', type=int, default=1, metavar='N',
                    help='defines size of ensemble or, by default, no ensemble if = 1')
parser.add_argument('-gbs', '--gen-batch-size', type=int, default=512, metavar='N',
                    help='batch size for descriptor generation (default: 512)')
parser.add_argument('-unsup', '--unsupervised', type=int, default=0,
                    help='unsupervised pretraining (1) as initial step or random weights (0)')

args = parser.parse_args()

# config
dataset = args.dataset
imagenetdir = args.imagenetdir
datadir = args.datadir
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
B = 10 # number of augmentation iterations (batches)
F = 5 # batch increment in %
bSize = 64000 # batch increment in images
#irs = [1, 100] # imbalance ratios instead of ir = args.imbalance_ratio
irs = [100]
msa = [512, 768] # descriptor lengths for multiscale aggregation
nr = 0.0 # not used in this project
ens = args.ensemble_size
smp = args.sample_steps
gbs = args.gen_batch_size
unsup = args.unsupervised
if unsup == 1:
    unsup_prefix = 'unsup_'
else:
    unsup_prefix = ''
# FORMAT: [imbalance ratio, reserved, subtype, augmentation method, descriptor length, ensemble size]
cfg = list()
for ir in irs:
    #cfg.append([ir, nr, 'entMC', 'topUncert', 0, ens])
    #cfg.append([ir, nr, 'varMC', 'topUncert', 0, ens])
    #cfg.append([ir, nr, 'stdMC', 'topUncert', 0, ens])
    #cfg.append([ir, nr, 'baldMC','topUncert', 0, ens])
    
    cfg.append([ir, nr, 'none',  'random',    0,  1])
    
    for dl in msa:
        cfg.append([ir, nr, 'none',   'topKandPcc',        dl, 1]) # feature similarity only: R_z
        cfg.append([ir, nr, 'gradAbl','topKandPccFprGrad', dl, 1]) # practical Fisher kernel (PFK) with identity FIM and true labels: R_{z,g}, S = y
        cfg.append([ir, nr, 'grad',   'topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, S = \hat{y}

        #cfg.append([ir, nr, 'gradFmap','topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, S = \hat{p}(y,z) (R_z for conditional estimate)
        #cfg.append([ir, nr, 'gradGmap','topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, S = \hat{p}(y,z) (R_{z,g} for conditional estimate)

# main loop
print(tabulate(cfg))
if args.initial:
    for r in range(args.run_start, args.run_stop):
        runFolder = 'run'+str(r)
        for ir in irs:
            for e in range(ens):
                print ('UNSUP (b=0) run # {} for ensemble # {}({}) with imbalance ratio = {}'.format(r, e, ens, ir))
                os.system('python3      unsup.py {} -ir {} -nr {} -r {} -ens {} -e {} --save-folder {}'.format(imagenetdir,
                    ir, nr, runFolder, ens, e, datadir))
else:
    for r in range(args.run_start, args.run_stop):
        runFolder = 'run'+str(r)
        # iterations
        for c in cfg:
            for b in range(0,B):
                if c[3] != 'random':
                    for e in range(ens):
                        print ('GEN DESCR (b={}) run # {} for ensemble # {}({}) using {} steps with imbalance ratio = {}: {}'.format(b, r, e, ens, smp, c[0], c))
                        os.system('python3  gen_descr.py {} -ir {} -nr {} -sub {} -aug {} -b {} -ts {}        -dl {} -r {} -ens {} -e {} -smp {} -bs {} --save-folder {} -unsup {}'.format(imagenetdir,
                            c[0], c[1], c[2], c[3], b, b*bSize,        c[4], runFolder, ens, e, smp, gbs, datadir, unsup))
                for e in range(ens):
                    print ('MAIN DESCR (b={}) run # {} for ensemble # {}({}) with imbalance ratio = {}: {}'.format(b, r, e, ens, c[0], c))
                    os.system('python3 main_descr.py {} -ir {} -nr {} -sub {} -aug {} -b {} -ts {} -as {} -dl {} -r {} -ens {} -e {}                  --save-folder {} -unsup {}'.format(imagenetdir,
                            c[0], c[1], c[2], c[3], b, b*bSize, bSize, c[4], runFolder, ens, e,           datadir, unsup))
