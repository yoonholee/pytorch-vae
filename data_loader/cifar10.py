from torchvision import datasets

class cifar10(datasets.CIFAR10):
    def get_mean_img(self):
        return self.train_data.mean(0)

