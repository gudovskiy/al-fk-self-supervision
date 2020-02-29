import argparse, os, torch
from tabulate import tabulate

parser = argparse.ArgumentParser(description='PyTorch MNIST Run')
parser.add_argument('--dataset', default='MNIST', type=str, help='dataset name')
parser.add_argument('--datadir', default='../local_data/MNIST', type=str, help='dir to save/read data')
parser.add_argument("--gpu", default='0', type=str, help='GPU device number')
parser.add_argument('--run-start', type=int, default=0, metavar='R', help='run start pointer (default: 0)')
parser.add_argument('--run-stop', type=int, default=10, metavar='R', help='run stop pointer (default: 10)')
parser.add_argument('--batch-size', type=int, default=25, help='Batch size used for training and testing')
parser.add_argument('-ir', '--imbalance-ratio', type=int, default=1, metavar='N', help='ratio of 0..4 to 5..9 labels in the training dataset drawn from uniform distribution')
parser.add_argument('--train-iterations', type=int, default=10000, help='Number of training iterations')
parser.add_argument('--latent-dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')
parser.add_argument('--adversary-param', type=int, default=1, help='Hyperparameter for training. lambda2 in the paper')

args = parser.parse_args()
# config
dataset = args.dataset
datadir = args.datadir
batch_size = args.batch_size
ir = args.imbalance_ratio
iters = args.train_iterations
zdim = args.latent_dim
lambda2 = args.adversary_param
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if not os.path.exists('./results/'):
    os.mkdir('./results/')

# main loop
for r in range(args.run_start, args.run_stop):
    runFolder = 'run'+str(r)
    outFolder = './results/' + dataset + '_run' + str(r) + '_ir_' + str(ir)
    if not os.path.exists(outFolder):
        os.mkdir(outFolder)
    #
    model_postfix = '{}batch_{}_ir_{}_nr_{}_sub_{}_aug_{}'.format('unsup_', 0, ir, 0.0, 'none', 'none')
    model_folder = '{}/{}/checkpoint'.format(datadir, runFolder)
    checkpoint_file = '{}/init_{}.pt'.format(model_folder, model_postfix)
    #
    print(runFolder, outFolder, checkpoint_file)
    # iterations
    os.system('python3 main.py --dataset {} --batch_size {} --train_iterations {} --latent_dim {} --adversary_param {} --data_path {} --out_path {} --imbalance_ratio {} --run_folder {} --pretrained {}'.format(
        dataset, batch_size, iters, zdim, lambda2, datadir, outFolder, ir, runFolder, checkpoint_file))
