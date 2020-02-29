import argparse, os, datetime, torch, pickle
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

parser = argparse.ArgumentParser(description='PyTorch ImageNet Run')
parser.add_argument('--dataset', default='ImageNet', type=str,
                    help='dataset name')
parser.add_argument('--datadir', default='../local_data/ImageNet', type=str,
                    help='dir to save/read data')
parser.add_argument('--run-start', type=int, default=0, metavar='R',
                    help='run start pointer (default: 0)')
parser.add_argument('--run-stop', type=int, default=1, metavar='R',
                    help='run stop pointer (default: 1)')
parser.add_argument('-ir', '--imbalance-ratio', type=int, default=1, metavar='N',
                    help='For half of the random classes we evict (ir-1)/ir labels chosen from SOME distribution to emulate distorted training dataset')

args = parser.parse_args()

# config
dataset = args.dataset
datadir = args.datadir
B = 10 # number of augmentation iterations (batches)
F = 5 # batch increment in %
bSize = 64000 # batch increment in images
msa = [512] # descriptor lengths for multiscale aggregation
K = 32
un = 1 # unsupervised or random
ens = [1,4] # ensemble_size
ir = args.imbalance_ratio
# FORMAT: [imbalance ratio, unsupervised, subtype, augmentation method, descriptor length, ensemble size]
cfg = list()
cfg.append([ir, un, 'none',  'random',    0,  1])
for e in ens:
    #cfg.append([ir, un, 'entMC', 'topUncert', 0, e])
    cfg.append([ir, un, 'varMC', 'topUncert', 0, e])
    #cfg.append([ir, un, 'stdMC', 'topUncert', 0, e])
    #cfg.append([ir, un, 'baldMC','topUncert', 0, e])
for dl in msa:
    cfg.append([ir, un, 'none',   'topKandPcc',        dl, 1]) # feature similarity only: R_z
    cfg.append([ir, un, 'gradAbl','topKandPccFprGrad', dl, 1]) # practical Fisher kernel (PFK) with identity FIM and true labels: R_{z,g}, S = y
    cfg.append([ir, un, 'grad',   'topKandPccFprGrad', 768, 1]) # PFK and pseudo labels: R_{z,g}, S = \hat{y}

    #cfg.append([ir, un, 'cor',     'topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, S = \tr(C_{z,g})
    #cfg.append([ir, un, 'amiFull', 'topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, S = I(z;g)
    #cfg.append([ir, un, 'amiDiag', 'topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, simplified S = I(z;g) with diagonals of cross-covariance

    cfg.append([ir, un, 'gradFmap', 'topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, S = \hat{p}(y,z) (R_z for conditional estimate)
    #cfg.append([ir, un, 'gradGmap', 'topKandPccFprGrad', dl, 1]) # PFK and pseudo labels: R_{z,g}, S = \hat{p}(y,z) (R_{z,g} for conditional estimate)

cfg.append([ir, un, 'gradFmap', 'topKandPccFprGrad', 768, 1])
#cfg.append([ir, un, 'gradFmap', 'topKandPccFprGrad', 768, 1])

# analyze results
print(tabulate(cfg))
results = cfg
for c in results:
    acc = np.zeros((args.run_stop-args.run_start, B))
    if c[1] == 1:
        unsup_prefix = 'unsup_'
    else:
        unsup_prefix = ''
    for b in range(0,B):
        model_postfix = '{}batch_{}_size_{}_ir_{}_nr_0.0_sub_{}_aug_{}_L_{}'.format(unsup_prefix, b+1, (b+1)*bSize, c[0], c[2], c[3], c[4])
        for r in range(0, args.run_stop-args.run_start):
            if c[5] > 1:
                ave = 0.0
                for e in range(c[5]):
                    checkpoint_file = '{}/{}/checkpoint/best_{}_E_{}.pt'.format(datadir, 'run'+str(args.run_start+r), model_postfix, e)
                    if os.path.isfile(checkpoint_file):
                        checkpoint = torch.load(checkpoint_file, map_location='cpu')
                        ave += checkpoint['acc']
                ave /= c[5] # average ensemble accuracy
                acc[r,b] = ave
            else:
                checkpoint_file = '{}/{}/checkpoint/best_{}.pt'.format(datadir, 'run'+str(args.run_start+r), model_postfix)
                if os.path.isfile(checkpoint_file):
                    checkpoint = torch.load(checkpoint_file, map_location='cpu')
                    acc[r,b] = checkpoint['acc']
    #
    c.append(np.round(acc*100.0)/100.0)
#
#results.append([ir, 0, 'none', 'VAAL', 'none', 1, np.load('../vaal/ImageNet_ir_{}.npy'.format(ir))])
# full-dataset baseline
if   ir == 1:
    results.append([1,   0, 'none', 'all train', 'none', 1, np.repeat([70.10], B)])
elif ir == 100:
    results.append([100, 0, 'none', 'all train', 'none', 1, np.repeat([38.40], B)])
else:
    print('Wrong IR!')
    sys.exit(0)
# print table
print(tabulate(results))
# save results
results_file = '{}_ir_{}_run_{}'.format(dataset, ir, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
with open(results_file+'.txt', 'w') as f:
    for item in results:
        f.write("%s\n" % item)
with open(results_file+'.pkl', 'wb') as f:
    pickle.dump(results, f)
# load results
#results_file = '{}_{}run_{}'.format(dataset, unsup_prefix, args.result_date)
with open(results_file+'.pkl', 'rb') as f:
    results = pickle.load(f)
#
pdf = PdfPages(results_file+'.pdf')
fontLegend = 11.5
fontAxis = 14
fontText = 14
k = ['-mx', '-bx', '-gx', '-rx', '-.md', '-.bd', '-.gd', '-.rd']
#k = ['-bx', '-md', '-yh', '-.gx', '-rx', '-c*']
n = [ ':cd',  ':md',  ':yd',  ':kd']
o = ['--ch', '--mh', '--yh', '--kh']
m = ['-.yo', '-yx']
plt.figure()
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'mathtext.fontset': 'cm'})

ax = plt.subplot(1, 1, 1)
results_list = results
x = F*np.array(range(1, (B+1), 1))
for i, r in enumerate(results_list):
    if r[6].ndim == 1: # baseline
        plt.plot(x, r[6], ':r', label=r[3], markersize=6, linewidth=1.5)
for i, r in enumerate(results_list):
    if r[3] == 'random':
        plt.errorbar(x, np.mean(r[6], axis=0), yerr=1*np.std(r[6], axis=0), fmt='--k', label=r[3], markersize=6, linewidth=1.5, elinewidth=1.0, capsize=4)
j = 0
for i, r in enumerate(results_list):
    if r[3] == 'VAAL':
        plt.errorbar(x, np.mean(r[6], axis=0), yerr=1*np.std(r[6], axis=0), fmt='-gD', label=r'%s' % (r[3]), markersize=6, linewidth=1.5, elinewidth=1.0, capsize=4)
        j = j + 1
j = 0
for i, r in enumerate(results_list):
    if (r[3] == 'topUncert') and (r[5] == 1):
        r[2] = 'varR' # rename for paper Fig.
        plt.errorbar(x, np.mean(r[6], axis=0), yerr=1*np.std(r[6], axis=0), fmt=n[j], label=r'%s, $E_{%d}, K_{%d}$' % (r[2], r[5], K/r[5]), markersize=6, linewidth=1.5, elinewidth=1.0, capsize=4)
        j = j + 1
j = 0
for i, r in enumerate(results_list):
    if (r[3] == 'topUncert') and (r[5] > 1):
        r[2] = 'varR' # rename for paper Fig.
        plt.errorbar(x, np.mean(r[6], axis=0), yerr=1*np.std(r[6], axis=0), fmt=o[j], label=r'%s, $E_{%d}, K_{%d}$' % (r[2], r[5], K/r[5]), markersize=6, linewidth=1.5, elinewidth=1.0, capsize=4)
        j = j + 1
j = 0
for i, r in enumerate(results_list):
    if (r[3] == 'topKandPcc'):
        L = str(r[4])
        plt.errorbar(x, np.mean(r[6], axis=0), yerr=1*np.std(r[6], axis=0), fmt=m[j], label=r'$R_z, L_{%d}$' % (r[4]), markersize=6, linewidth=1.5, elinewidth=1.0, capsize=4)
        j = j + 1
j = 0
for i, r in enumerate(results_list):
    if (r[3] == 'topKandPccFprGrad'):
        L = str(r[4])
        if   r[2] == 'gradAbl':
            plt.errorbar(x, np.mean(r[6], axis=0), yerr=1*np.std(r[6], axis=0), fmt=k[j], label=r'$R_{z,g}, S=y, L_{%d}$' % (r[4]), markersize=6, linewidth=1.5, elinewidth=1.0, capsize=4)
        elif r[2] == 'grad':
            plt.errorbar(x, np.mean(r[6], axis=0), yerr=1*np.std(r[6], axis=0), fmt=k[j], label=r'$R_{z,g}, S=\hat{y}, L_{%d}$' % (r[4]), markersize=6, linewidth=1.5, elinewidth=1.0, capsize=4)
        elif r[2] == 'cor':
            plt.errorbar(x, np.mean(r[6], axis=0), yerr=1*np.std(r[6], axis=0), fmt=k[j], label=r'$R_{z,g}, S=\mathrm{tr}(C_{z,g}), L_{%d}, K_{%d}$' % (r[4], K), markersize=6, linewidth=1.5, elinewidth=1.0, capsize=4)
        elif r[2] == 'amiFull':
            plt.errorbar(x, np.mean(r[6], axis=0), yerr=1*np.std(r[6], axis=0), fmt=k[j], label=r'$R_{z,g}, S=I(z;g), L_{%d}, K_{%d}$' % (r[4], K), markersize=6, linewidth=1.5, elinewidth=1.0, capsize=4)
        elif r[2] == 'gradFmap' and r[1] == 0:
            plt.errorbar(x, np.mean(r[6], axis=0), yerr=1*np.std(r[6], axis=0), fmt=k[j], label=r'$R_{z,g}, S=\hat{p}(y,z), L_{%d}, \theta^0_{rnd}$' % (r[4]), markersize=6, linewidth=1.5, elinewidth=1.0, capsize=4)
        elif r[2] == 'gradFmap' and r[1] == 1:
            plt.errorbar(x, np.mean(r[6], axis=0), yerr=1*np.std(r[6], axis=0), fmt=k[j], label=r'$R_{z,g}, S=\hat{p}(y,z), L_{%d} (\mathbf{ours})$' % (r[4]), markersize=6, linewidth=1.5, elinewidth=1.0, capsize=4)
        elif r[2] == 'gradGmap' and r[1] == 0:
            plt.errorbar(x, np.mean(r[6], axis=0), yerr=1*np.std(r[6], axis=0), fmt=k[j], label=r'$R_{z,g}, S=\hat{p}_{z,g}(y,z), L_{%d}, \theta^0_{rnd}$' % (r[4]), markersize=6, linewidth=1.5, elinewidth=1.0, capsize=4)
        elif r[2] == 'gradGmap' and r[1] == 1:
            plt.errorbar(x, np.mean(r[6], axis=0), yerr=1*np.std(r[6], axis=0), fmt=k[j], label=r'$R_{z,g}, S=\hat{p}_{z,g}(y,z), L_{%d}$' % (r[4]), markersize=6, linewidth=1.5, elinewidth=1.0, capsize=4)
        else:
            print('Wrong Label Estimation Type!')
            sys.exit(0)
        #
        j = j + 1
#
if   ir == 1:
    plt.axis([F, F*B, 42.0, 72.0])
    plt.yticks(np.arange(42.0, 72.0, 3.0))
elif ir == 100:
    plt.axis([F, F*B, 26.0, 41.0])
    plt.yticks(np.arange(26.0, 41.0, 1.5))
else:
    print('Wrong IR!')
    sys.exit(0)
#
if ir == 1:
    plt.annotate("",
                xy=(22.5, 60.5), xycoords='data',
                xytext=(22.5, 62.5), textcoords='data',
                arrowprops=dict(arrowstyle="<|-|>",
                connectionstyle="arc3", color='black', linewidth=0.5), size=12)

    plt.annotate("",
                xy=(37.5, 66.5), xycoords='data',
                xytext=(37.5, 71.0), textcoords='data',
                arrowprops=dict(arrowstyle="<|-|>",
                connectionstyle="arc3", color='black', linewidth=0.5), size=12)

    plt.text(20.5, 63.0, r'$1.5\%$', rotation = 0, fontsize = 12)
    plt.text(38, 67.5, r'$5\%$', rotation = 0, fontsize = 12)
    plt.text(6.0, 67.0, '(c)', rotation = 0, fontsize = 16)
elif ir == 100:
    plt.annotate("",
                xy=(22.5, 36.0), xycoords='data',
                xytext=(22.5, 38.0), textcoords='data',
                arrowprops=dict(arrowstyle="<|-|>",
                connectionstyle="arc3", color='black', linewidth=0.5), size=12)
    plt.annotate("",
                xy=(22.25, 38.0), xycoords='data',
                xytext=(39.0, 38.0), textcoords='data',
                arrowprops=dict(arrowstyle="-|>",
                connectionstyle="arc3", color='black', linewidth=0.5), size=12)

    plt.annotate("",
                xy=(37.5, 38.8), xycoords='data',
                xytext=(37.5, 40.3), textcoords='data',
                arrowprops=dict(arrowstyle="<|-|>",
                connectionstyle="arc3", color='black', linewidth=0.5), size=12)

    plt.text(22.8, 36.7, r'$2\%$', rotation = 0, fontsize = 12)
    plt.text(23.7, 38.6, r'$-42\%$', rotation = 0, fontsize = 12)
    plt.text(37.7, 39.3, r'$1.5\%$', rotation = 0, fontsize = 12)
    plt.text(6.0, 40.0, '(d)', rotation = 0, fontsize = 16)
#
plt.xticks(np.arange(F, F*(B+1), F))
plt.legend(loc='lower right', shadow=False, fontsize = fontLegend, ncol=2)
plt.xlabel('Fraction of full training dataset, %', fontsize = fontAxis)
plt.ylabel('Top-1 accuracy, %', fontsize = fontAxis)
plt.grid(True)
#
fig = plt.gcf()
fig.set_size_inches(6, 5, forward=True)
plt.tight_layout()
plt.tight_layout()
plt.tight_layout()
#plt.show()
plt.savefig(results_file+'.png', bbox_inches='tight')
pdf.savefig(bbox_inches='tight')
pdf.close()
