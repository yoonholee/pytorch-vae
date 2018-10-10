import torch
from torchvision import datasets, transforms

def data_loaders(args):
    if args.dataset == 'mnist':
        loader_fn, root = datasets.MNIST, './dataset/mnist'
    elif args.dataset == 'fashionmnist':
        loader_fn, root = datasets.FashionMNIST, './dataset/fashionmnist'

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        loader_fn(root, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        loader_fn(root, train=False, download=True, transform=transforms.ToTensor()),
        batch_size=args.test_batch_size, shuffle=False, **kwargs) # need high bs to calculate L_5000.
    return train_loader, test_loader

