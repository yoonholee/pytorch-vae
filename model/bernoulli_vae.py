import numpy as np
import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal

from .vae_base import VAE


class BernoulliVAE(VAE):
    def __init__(self, device, img_shape, h_dim, z_dim, analytic_kl, mean_img):
        super().__init__(device, z_dim, analytic_kl)
        x_dim = np.prod(img_shape)
        self.img_shape = img_shape
        self.proc_data = lambda x: x.to(device).reshape(-1, x_dim)
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, h_dim), nn.Tanh(), nn.Linear(h_dim, h_dim), nn.Tanh()
        )
        self.enc_mu = nn.Linear(h_dim, z_dim)
        self.enc_sig = nn.Linear(h_dim, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.Tanh(),
            nn.Linear(h_dim, h_dim), nn.Tanh(),
            nn.Linear(h_dim, x_dim),
        )  # using Bern(logit) is equivalent to putting sigmoid here.

        self.apply(self.init)
        mean_img = np.clip(mean_img, 1e-8, 1.0 - 1e-7)
        mean_img_logit = np.log(mean_img / (1.0 - mean_img))
        self.decoder[-1].bias = torch.nn.Parameter(torch.Tensor(mean_img_logit))

    def init(self, module):
        if type(module) == nn.Linear:
            torch.nn.init.xavier_uniform_(
                module.weight, gain=nn.init.calculate_gain("tanh")
            )
            module.bias.data.fill_(0.01)

    def encode(self, x):
        x = self.proc_data(x)
        h = self.encoder(x)
        mu, _std = self.enc_mu(h), self.enc_sig(h)
        return Normal(mu, nn.functional.softplus(_std))  # torch.exp(.5 * _std)

    def decode(self, z):
        x = self.decoder(z)
        return Bernoulli(logits=x)

    def lpxz(self, true_x, x_dist):
        return x_dist.log_prob(true_x).sum(-1)

    def sample(self, num_samples=64):
        z = self.prior.sample((num_samples,))
        x_dist = self.decode(z)
        return x_dist.sample().view(num_samples, *self.img_shape)
