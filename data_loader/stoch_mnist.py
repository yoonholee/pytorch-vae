import torch
from torchvision import datasets, transforms
from PIL import Image


class stochMNIST(datasets.MNIST):
    """ Gets a new stochastic binarization of MNIST at each call. """
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        img = Image.fromarray(img.numpy(), mode='L')
        img = transforms.ToTensor()(img)
        img = torch.bernoulli(img)  # stochastically binarize
        return img, target

    def get_mean_img(self):
        imgs = self.train_data.type(torch.float) / 255
        mean_img = imgs.mean(0).reshape(-1).numpy()
        return mean_img
