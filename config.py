import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--eval', type=bool, default=False)
parser.add_argument('--figs', action='store_true')

parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128) # iwae uses 20
parser.add_argument('--importance_num', type=int, default=1) # k of iwae. Use 1 for vanilla vae
parser.add_argument('--epochs', type=int, default=100000) # iwae uses 3280

analytic_kl_parser = parser.add_mutually_exclusive_group(required=False)
analytic_kl_parser.add_argument('--analytic_kl', dest='analytic_kl', action='store_true')
analytic_kl_parser.add_argument('--no_analytic_kl', dest='analytic_kl', action='store_false')
parser.set_defaults(analytic_kl=True)
parser.add_argument('--h_dim', type=int, default=200)
parser.add_argument('--z_dim', type=int, default=50)
parser.add_argument('--beta', type=float, default=1)

def get_args():
    args = parser.parse_args()
    if args.dataset == 'mnist':
        args.x_dim = 784
    args.exp_name = 'k{}_h{}_z{}_lr{}_beta{}'.format(
        args.importance_num, args.h_dim, args.z_dim, args.learning_rate, args.beta)
    args.figs_dir = 'figs/{}'.format(args.exp_name)
    args.out_dir = 'result/{}'.format(args.exp_name)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.figs_dir):
        os.makedirs(args.figs_dir)
    return args

