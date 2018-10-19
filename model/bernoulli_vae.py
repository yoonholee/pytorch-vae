import numpy as np
import torch
from torch import nn
from torch.distributions.bernoulli import Bernoulli
from .vae_base import VAE

class BernoulliVAE(VAE):
    def __init__(self, device, x_dim, h_dim, z_dim, beta, analytic_kl, mean_img):
        VAE.__init__(self, device, x_dim, h_dim, z_dim, beta, analytic_kl, mean_img)
        self.proc_data = lambda x: x.to(device).reshape(-1, x_dim)
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, h_dim), nn.Tanh(),
            nn.Linear(h_dim, h_dim), nn.Tanh())
        self.enc_mu = nn.Linear(h_dim, z_dim)
        self.enc_sig = nn.Linear(h_dim, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.Tanh(),
            nn.Linear(h_dim, h_dim), nn.Tanh(),
            nn.Linear(h_dim, x_dim)) # using Bern(logit) is equivalent to putting sigmoid here.

        self.apply(self.init)
        mean_img = np.clip(mean_img, 1e-8, 1. - 1e-7)
        mean_img_logit = np.log(mean_img / (1. - mean_img))
        self.decoder[-1].bias = torch.nn.Parameter(torch.Tensor(mean_img_logit))

    def decode(self, z):
        x = self.decoder(z)
        return Bernoulli(logits=x)

