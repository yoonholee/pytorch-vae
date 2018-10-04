import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

class VAE(nn.Module):
    def __init__(self, device, x_dim, h_dim, z_dim, beta):
        super(VAE, self).__init__()
        self.beta = beta
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.enc_fc1 = nn.Linear(x_dim, h_dim)
        self.enc_fc2_mu = nn.Linear(h_dim, z_dim)
        self.enc_fc2_sig = nn.Linear(h_dim, z_dim)
        self.dec_fc1 = nn.Linear(z_dim, h_dim)
        self.dec_fc2 = nn.Linear(h_dim, x_dim)
        self.prior = Normal(torch.zeros([z_dim]).to(device),
                            torch.ones([z_dim]).to(device))
        self.analytic_kl = True

    def encode(self, x):
        h1 = F.relu(self.enc_fc1(x))
        mu, _std = self.enc_fc2_mu(h1), self.enc_fc2_sig(h1)
        std = F.softplus(_std)  # some implementations use exp(_std/2)
        z_dist = Normal(mu, std)
        return mu, z_dist

    def decode(self, z):
        h3 = F.relu(self.dec_fc1(z))
        x = self.dec_fc2(h3)
        x_dist = Bernoulli(logits=x)
        return x_dist

    def forward(self, true_x):
        true_x = true_x.view(-1, self.x_dim)
        mu, z_dist = self.encode(true_x)
        z = z_dist.rsample() if self.training else mu
        x_dist = self.decode(z)
        return {'x_dist': x_dist, 'z': z, 'z_dist': z_dist}

    def loss(self, true_x, z, x_dist, z_dist):
        if not self.analytic_kl:
            # SGVB^A: log p(z) + log p(x|z)- log q(z|x)
            lpz = self.prior.log_prob(z).sum(1)
            lpxz = x_dist.log_prob(true_x).sum(1) # equivalent to BCE(x_dist.logits, true_x)
            lqzx = z_dist.log_prob(z).sum(1)
            elbo = lpz + lpxz - lqzx
            loss = -elbo

        if self.analytic_kl:
            # SGVB^B: -KL(q(z|x)||p(z)) + log p(x|z). Use when KL can be done analytically.
            kl = torch.distributions.kl.kl_divergence(z_dist, self.prior).sum(1)
            lpxz = x_dist.log_prob(true_x).sum(1) # equivalent to BCE(x_dist.logits, true_x)
            elbo = -kl + lpxz
            loss = self.beta * kl - lpxz #betaVAE. for original VAE simply set beta=1

        return loss.mean(), elbo.mean()

    def sample(self, num_samples=64):
        z = self.prior.sample((num_samples,))
        sample = self.decode(z).probs
        return sample.view(num_samples, 1, 28, 28)

