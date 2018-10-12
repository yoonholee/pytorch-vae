import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
import urllib.request

class binaryMNIST(data.Dataset):
    """ Binarized MNIST dataset, proposed in
    http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf """
    train_file = 'binarized_mnist_train.amat'
    test_file = 'binarized_mnist_test.amat'
    val_file = 'binarized_mnist_valid.amat'

    def __init__(self, root, train=True, transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set

        if download: self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.data = self._get_data(train=train)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img, mode='F')
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(-1) # Meaningless tensor instead of target

    def __len__(self):
        return len(self.data)

    def _get_data(self, train=True):
        def filename_to_np(filename):
            with open(filename) as f:
                lines = f.readlines()
            return np.array([[int(i)for i in line.split()] for line in lines]).astype('float32')

        if train:
            data = np.concatenate([filename_to_np(os.path.join(self.root, self.train_file)),
                                        filename_to_np(os.path.join(self.root, self.val_file))])
        else:
            data = filename_to_np(os.path.join(self.root, self.val_file))
        return data.reshape(-1, 28, 28)

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading binary MNIST...')
        for dataset in ['train', 'valid', 'test']:
            filename = 'binarized_mnist_{}.amat'.format(dataset)
            url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(dataset)
            print('Downloading from {}...'.format(url))
            local_filename = os.path.join(self.root, filename)
            urllib.request.urlretrieve(url, local_filename)
            print('Saved to {}'.format(local_filename))
        print('Done!')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.train_file))

def data_loaders(args):
    if args.dataset == 'binarymnist':
        loader_fn, root = binaryMNIST, './dataset/binarymnist'
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

