import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import Image
import urllib.request
import h5py
import scipy.io

class omniglot(data.Dataset):
    """ omniglot dataset """
    url = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'

    def __init__(self, root, train=True, transform=None, download=False):
        # we ignore transform.
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        if download: self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')

        self.data = self._get_data(train=train)

    def __getitem__(self, index):
        img = self.data[index].reshape(28, 28)
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img).type(torch.FloatTensor)
        img = torch.bernoulli(img) # stochastically binarize
        return img, torch.tensor(-1) # Meaningless tensor instead of target

    def __len__(self):
        return len(self.data)

    def _get_data(self, train=True):
        def reshape_data(data):
            return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')

        omni_raw = scipy.io.loadmat(os.path.join(self.root, 'chardata.mat'))
        data_str = 'data' if train else 'testdata'
        data = reshape_data(omni_raw[data_str].T.astype('float32'))
        return data

    def download(self):
        if self._check_exists():
            return
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        print('Downloading from {}...'.format(self.url))
        local_filename = os.path.join(self.root, 'chardata.mat')
        urllib.request.urlretrieve(self.url, local_filename)
        print('Saved to {}'.format(local_filename))

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'chardata.mat'))


class binaryMNIST(data.Dataset):
    """ Binarized MNIST dataset, proposed in
    http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf """
    train_file = 'binarized_mnist_train.amat'
    val_file = 'binarized_mnist_valid.amat'
    test_file = 'binarized_mnist_test.amat'

    def __init__(self, root, train=True, transform=None, download=False):
        # we ignore transform.
        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set

        if download: self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        self.data = self._get_data(train=train)

    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        img = transforms.ToTensor()(img).type(torch.FloatTensor)
        return img, torch.tensor(-1) # Meaningless tensor instead of target

    def __len__(self):
        return len(self.data)

    def _get_data(self, train=True):
        with h5py.File(os.path.join(self.root, 'data.h5'), 'r') as hf:
            data = hf.get('train' if train else 'test')
            data = np.array(data)
        return data

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

        def filename_to_np(filename):
            with open(filename) as f:
                lines = f.readlines()
            return np.array([[int(i)for i in line.split()] for line in lines]).astype('int8')

        train_data = np.concatenate([filename_to_np(os.path.join(self.root, self.train_file)),
                                        filename_to_np(os.path.join(self.root, self.val_file))])
        test_data = filename_to_np(os.path.join(self.root, self.val_file))
        with h5py.File(os.path.join(self.root, 'data.h5'), 'w') as hf:
            hf.create_dataset('train', data=train_data.reshape(-1, 28, 28))
            hf.create_dataset('test', data=test_data.reshape(-1, 28, 28))
        print('Done!')

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'data.h5'))

def data_loaders(args):
    if args.dataset == 'omniglot':
        loader_fn, root = omniglot, './dataset/omniglot'
    elif args.dataset == 'fixedbinarymnist':
        loader_fn, root = binaryMNIST, './dataset/binarymnist'
    elif args.dataset == 'mnist':
        loader_fn, root = datasets.MNIST, './dataset/mnist'
    elif args.dataset == 'fashionmnist':
        loader_fn, root = datasets.FashionMNIST, './dataset/fashionmnist'

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        loader_fn(root, train=True, download=True, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        loader_fn(root, train=False, download=True, transform=transforms.ToTensor()),
        batch_size=args.test_batch_size, shuffle=False, **kwargs) # need high bs to calculate L_5000.
    return train_loader, test_loader

