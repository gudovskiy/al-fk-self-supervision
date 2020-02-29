import argparse, os, torch
from tabulate import tabulate

parser = argparse.ArgumentParser(description='PyTorch MNIST Run')
parser.add_argument('--dataset', default='MNIST', type=str,
                    help='dataset name')
parser.add_argument('--datadir', default='../local_data/MNIST', type=str,
                    help='dir to save/read data')
parser.add_argument("--gpu", default='0', type=str,
                    help='GPU device number')
parser.add_argument('--initial', action='store_true',
                    help='initial batch to generate starting models')
parser.add_argument('--run-start', type=int, default=0, metavar='R',
                    help='run start pointer (default: 0)')
parser.add_argument('--run-stop', type=int, default=10, metavar='R',
                    help='run stop pointer (default: 10)')
parser.add_argument('-smp', '--sample-steps', type=int, default=128, metavar='N',
                    help='number of samples for estimation (default: 128)')
parser.add_argument('-ens', '--ensemble-size', type=int, default=1, metavar='N',
                    help='defines size of ensemble or, by default, no ensemble if = 1')
parser.add_argument('-gbs', '--gen-batch-size', type=int, default=10000, metavar='N',
                    help='batch size for descriptor generation (default: 10000)')
parser.add_argument('-unsup', '--unsupervised', type=int, default=0,
                    help='unsupervised pretraining (1) as initial step or random weights (0)')

args = parser.parse_args()

# config
dataset = args.dataset
datadir = args.datadir
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
B = 10 # number of augmentation iterations (batches)
F = 0.25 # batch increment in %
bSize = 125 # batch increment in images
irs = [1,100] # imbalance ratios instead of ir = args.imbalance_ratio
#irs = [100]
msa = [80,20] # descriptor lengths for multiscale aggregation
#msa = [80]
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
for ir in irs: # we add "MC" to uncertainty methods here, while in the paper we use "MC" for pseudo-labeling only. Hence, the code has slightly different naming
    #cfg.append([ir, nr, 'entMC', 'topUncert', 0, ens])
    cfg.append([ir, nr, 'varMC', 'topUncert', 0, ens])
    #cfg.append([ir, nr, 'stdMC', 'topUncert', 0, ens])
    #cfg.append([ir, nr, 'baldMC','topUncert', 0, ens])
    
    cfg.append([ir, nr, 'none',  'random',    0,  1])
    
    for dl in msa:
        cfg.append([ir, nr, 'none',   'topKandPcc',        dl, 1]) # feature similarity only: R_z
        cfg.append([ir, nr, 'gradAbl','topKandPccFprGrad', dl, 1]) # practical Fisher kernel (PFK) with identity FIM and true labels: R_{z,g}, S = y
        cfg.append([ir, nr, 'grad',   'topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, S = \hat{y}

        #cfg.append([ir, nr, 'cor',    'topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, S = \tr(C_{z,g})
        #cfg.append([ir, nr, 'amiFull', 'topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, S = I(z;g)
        #cfg.append([ir, nr, 'amiDiag', 'topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, simplified S = I(z;g) with diagonals of cross-covariance matrices

        cfg.append([ir, nr, 'gradFmap','topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, S = \hat{p}(y,z) (R_z for conditional estimate)
        #cfg.append([ir, nr, 'gradGmap','topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, S = \hat{p}(y,z) (R_{z,g} for conditional estimate)

# main loop
print(tabulate(cfg))
if args.initial:
    for r in range(args.run_start, args.run_stop):
        runFolder = 'run'+str(r)
        for ir in irs:
            for e in range(ens):
                print ('UNSUP (b=0) run # {} for ensemble # {}({}) with imbalance ratio = {}'.format(r, e, ens, ir))
                os.system('python3      unsup.py -ir {} -nr {} -r {} -ens {} -e {} --save-folder {}'.format(
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
                        os.system('python3  gen_descr.py -ir {} -nr {} -sub {} -aug {} -b {} -ts {}        -dl {} -r {} -ens {} -e {} -smp {} -gbs {} --save-folder {} -unsup {}'.format(
                            c[0], c[1], c[2], c[3], b, b*bSize,        c[4], runFolder, ens, e, smp, gbs, datadir, unsup))
                for e in range(ens):
                    print ('MAIN DESCR (b={}) run # {} for ensemble # {}({}) with imbalance ratio = {}: {}'.format(b, r, e, ens, c[0], c))
                    os.system('python3 main_descr.py -ir {} -nr {} -sub {} -aug {} -b {} -ts {} -as {} -dl {} -r {} -ens {} -e {}                  --save-folder {} -unsup {}'.format(
                            c[0], c[1], c[2], c[3], b, b*bSize, bSize, c[4], runFolder, ens, e,           datadir, unsup))

