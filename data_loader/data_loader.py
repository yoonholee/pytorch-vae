import torch
from torchvision import transforms
from .stoch_mnist import stochMNIST
from .omniglot import omniglot
from .fixed_mnist import fixedMNIST
from .cifar10 import cifar10


def data_loaders(args):
    if args.dataset == 'omniglot':
        loader_fn, root = omniglot, './dataset/omniglot'
    elif args.dataset == 'fixedmnist':
        loader_fn, root = fixedMNIST, './dataset/fixedmnist'
    elif args.dataset == 'stochmnist':
        loader_fn, root = stochMNIST, './dataset/stochmnist'
    elif args.dataset == 'cifar10':
        loader_fn, root = cifar10, './dataset/cifar10'

    if args.dataset_dir != '': root = args.dataset_dir
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        loader_fn(root, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(  # need test bs <=64 to make L_5000 tractable in one pass
        loader_fn(root, train=False, download=True, transform=transforms.ToTensor()),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)
    return train_loader, test_loader
