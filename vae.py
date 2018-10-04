import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

class VAE(nn.Module):
    def __init__(self, device, x_dim, h_dim, z_dim, beta):
        super(VAE, self).__init__()
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, h_dim), nn.Tanh(),
            nn.Linear(h_dim, h_dim), nn.Tanh())
        self.enc_mu = nn.Linear(h_dim, z_dim)
        self.enc_sig = nn.Linear(h_dim, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.Tanh(),
            nn.Linear(h_dim, h_dim), nn.Tanh(),
            nn.Linear(h_dim, x_dim)) # using Bern(logit) is equivalent to putting sigmoid here.
        self.prior = Normal(torch.zeros([z_dim]).to(device),
                            torch.ones([z_dim]).to(device))
        self.analytic_kl = True

    def encode(self, x):
        h = self.encoder(x)
        mu, _std = self.enc_mu(h), self.enc_sig(h)
        std = torch.exp(.5 * _std)
        return Normal(mu, std)

    def decode(self, z):
        x = self.decoder(z)
        return Bernoulli(logits=x)

    def forward(self, true_x):
        z_dist = self.encode(true_x)
        #z = z_dist.rsample() if self.training else z_dist.loc
        z = z_dist.rsample()
        x_dist = self.decode(z)
        return {'x_dist': x_dist, 'z': z, 'z_dist': z_dist}

    def loss(self, true_x, z, x_dist, z_dist):
        if not self.analytic_kl: # SGVB^A: log p(z) + log p(x|z)- log q(z|x)
            lpz = self.prior.log_prob(z).sum(1)
            lpxz = x_dist.log_prob(true_x).sum(1) # equivalent to BCE(x_dist.logits, true_x)
            lqzx = z_dist.log_prob(z).sum(1)
            elbo = lpz + lpxz - lqzx
            loss = -elbo

        if self.analytic_kl: # SGVB^B: -KL(q(z|x)||p(z)) + log p(x|z). Use when KL can be done analytically.
            kl = torch.distributions.kl.kl_divergence(z_dist, self.prior).sum(1)
            lpxz = x_dist.log_prob(true_x).sum(1) # equivalent to BCE(x_dist.logits, true_x)
            elbo = -kl + lpxz
            loss = self.beta * kl - lpxz #betaVAE. for original VAE simply set beta=1

        return loss.mean(), elbo.mean()

    def sample(self, num_samples=64):
        z = self.prior.sample((num_samples,))
        sample = self.decode(z).probs
        return sample.view(num_samples, 1, 28, 28)

