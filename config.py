import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_interval', type=int, default=500)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--figs', action='store_true')

parser.add_argument('--dataset_dir', type=str, default='')
parser.add_argument('--dataset', type=str, default='stochmnist',
                    choices=['stochmnist', 'omniglot', 'fixedmnist'])
parser.add_argument('--batch_size', type=int, default=20) # iwae uses 20
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--mean_num', type=int, default=1) # M in "tighter variational bounds...". Use 1 for vanilla vae
parser.add_argument('--importance_num', type=int, default=1) # k of iwae. Use 1 for vanilla vae
parser.add_argument('--epochs', type=int, default=4000) # iwae uses 3280
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--no_iwae_lr', action='store_true')

parser.add_argument('--analytic_kl', action='store_true')
parser.add_argument('--h_dim', type=int, default=200)
parser.add_argument('--z_dim', type=int, default=50)
parser.add_argument('--beta', type=float, default=1)

def get_args():
    args = parser.parse_args()

    args.exp_name = 'm{}_k{}'.format(args.mean_num, args.importance_num)
    if args.dataset != 'stochmnist': args.exp_name += '_{}'.format(args.dataset)
    if args.seed != 42: args.exp_name += '_seed{}'.format(args.seed)
    if args.batch_size != 20: args.exp_name += '_bs{}'.format(args.batch_size)
    if args.h_dim != 200: args.exp_name += '_h{}'.format(args.h_dim)
    if args.z_dim != 50: args.exp_name += '_z{}'.format(args.z_dim)
    if args.learning_rate != 1e-3: args.exp_name += '_z{}'.format(args.learning_rate)
    if args.beta != 1: args.exp_name += '_beta{}'.format(args.beta)
    if args.analytic_kl: args.exp_name += '_analytic'
    if args.no_iwae_lr: args.exp_name += '_noiwaelr'

    args.figs_dir = os.path.join('figs', args.exp_name)
    args.out_dir = os.path.join('result', args.exp_name)
    args.best_model_file = os.path.join('result', args.exp_name, 'best_model.pt')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.figs_dir):
        os.makedirs(args.figs_dir)

    if args.dataset in ['fixedmnist', 'stochmnist', 'omniglot']:
        args.x_dim = 784
    return args

