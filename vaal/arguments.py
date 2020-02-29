import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Name of the dataset used.')
    parser.add_argument('--batch_size', type=int, default=25, help='Batch size used for training and testing')
    parser.add_argument('--train_iterations', type=int, default=20000, help='Number of training iterations')
    parser.add_argument('--latent_dim', type=int, default=32, help='The dimensionality of the VAE latent dimension')
    parser.add_argument('--data_path', type=str, default='/home/demo/data/MNIST', help='Path to where the data is')
    parser.add_argument('--beta', type=float, default=1, help='Hyperparameter for training. The parameter for VAE')
    parser.add_argument('--num_adv_steps', type=int, default=2, help='Number of adversary steps taken for every task model step')
    parser.add_argument('--num_vae_steps', type=int, default=2, help='Number of VAE steps taken for every task model step')
    parser.add_argument('--adversary_param', type=float, default=1, help='Hyperparameter for training. lambda2 in the paper')
    parser.add_argument('--out_path', type=str, default='./results', help='Path to where the output log will be')
    parser.add_argument('--log_name', type=str, default='accuracies.log', help='Final performance of the models will be saved with this name')
    ########################
    parser.add_argument('--imbalance_ratio', type=int, default=1, metavar='N', help='ratio of 0..4 to 5..9 labels in the training dataset drawn from uniform distribution')
    parser.add_argument('--run_folder', default='run99', type=str, help='dir to save run')
    parser.add_argument('--pretrained', default='', type=str, help='pretrained model')
    ########################

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    
    return args
