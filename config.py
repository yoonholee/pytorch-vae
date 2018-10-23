import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_interval', type=int, default=500)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--figs', action='store_true')
parser.add_argument('--arch', type=str, default='bernoulli', choices=['bernoulli', 'conv'])

parser.add_argument('--dataset_dir', type=str, default='')
parser.add_argument('--dataset', type=str, default='stochmnist',
                    choices=['stochmnist', 'omniglot', 'fixedmnist', 'cifar10'])
parser.add_argument('--batch_size', type=int, default=20)  # iwae uses 20
parser.add_argument('--test_batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=3280)  # iwae uses 3280

parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--no_iwae_lr', action='store_true')
parser.add_argument('--mean_num', type=int, default=1)  # M in "tighter variational bounds...". Use 1 for vanilla vae
parser.add_argument('--importance_num', type=int, default=1)  # k of iwae. Use 1 for vanilla vae
parser.add_argument('--analytic_kl', action='store_true')
parser.add_argument('--h_dim', type=int, default=200)
parser.add_argument('--z_dim', type=int, default=50)


def get_args():
    args = parser.parse_args()

    def cstr(arg, arg_name, default, custom_str=False):
        """ Get config str for arg, ignoring if set to default. """
        not_default = arg != default
        if not custom_str:
            custom_str = f'_{arg_name}{arg}'
        return custom_str if not_default else ''

    args.exp_name = (f'm{args.mean_num}_k{args.importance_num}'
                     f'{cstr(args.dataset, "", "stochmnist")}{cstr(args.arch, "", "bernoulli")}'
                     f'{cstr(args.seed, "seed", 42)}{cstr(args.batch_size, "bs", 20)}'
                     f'{cstr(args.h_dim, "h", 200)}{cstr(args.z_dim, "z", 50)}'
                     f'{cstr(args.learning_rate, "lr", 1e-3)}{cstr(args.analytic_kl, None, False, "_analytic")}'
                     f'{cstr(args.no_iwae_lr, None, False, "_noiwae")}{cstr(args.epochs, "epoch", 3280)}')

    args.figs_dir = os.path.join('figs', args.exp_name)
    args.out_dir = os.path.join('result', args.exp_name)
    args.best_model_file = os.path.join('result', args.exp_name, 'best_model.pt')
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.figs_dir):
        os.makedirs(args.figs_dir)

    args.log_likelihood_k = 100 if args.dataset == 'cifar10' else 5000
    args.x_dim = 32*32 if args.dataset == 'cifar10' else 28*28
    #TODO: something like args.img_size here
    return args
