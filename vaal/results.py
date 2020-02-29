import argparse, os, torch
import numpy as np
from tabulate import tabulate

parser = argparse.ArgumentParser(description='PyTorch MNIST Run')
parser.add_argument('--dataset', default='MNIST', type=str, help='dataset name')
parser.add_argument('--run-start', type=int, default=0, metavar='R', help='run start pointer (default: 0)')
parser.add_argument('--run-stop', type=int, default=10, metavar='R', help='run stop pointer (default: 10)')
parser.add_argument('-ir', '--imbalance-ratio', type=int, default=1, metavar='N', help='ratio of 0..4 to 5..9 labels in the training dataset drawn from uniform distribution')

args = parser.parse_args()
# config
dataset = args.dataset
ir = args.imbalance_ratio

# main loop
B = 10
results = np.zeros((args.run_stop-args.run_start, B))
results_file = dataset + '_ir_' + str(ir) + '.npy'
for r in range(args.run_start, args.run_stop):
    outFolder = './results/' + dataset + '_run' + str(r) + '_ir_' + str(ir)
    results[r] = np.round(np.asarray(torch.load('{}/{}'.format(outFolder, 'accuracies.log'))), decimals=2)
    #print(outFolder, results)
#print(len(results), results)
np.save(results_file, results)
print(tabulate(results))
