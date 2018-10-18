import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

class VAE(nn.Module):
    def __init__(self, device, x_dim, h_dim, z_dim, beta, analytic_kl, mean_img):
        super(VAE, self).__init__()
        self.train_step = 0
        self.best_loss = np.inf
        self.proc_data = lambda x: x.to(device).reshape(-1, x_dim)
        self.beta = beta
        self.analytic_kl = analytic_kl
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
        def init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                m.bias.data.fill_(.01)
        self.apply(init)
        mean_img = np.clip(mean_img, 1e-8, 1. - 1e-7)
        mean_img_logit = np.log(mean_img / (1. - mean_img))
        self.decoder[-1].bias = torch.nn.Parameter(torch.Tensor(mean_img_logit))

    def encode(self, x):
        x = self.proc_data(x)
        h = self.encoder(x)
        mu, _std = self.enc_mu(h), self.enc_sig(h)
        std = nn.functional.softplus(_std) #std = torch.exp(.5 * _std)
        return Normal(mu, std)

    def decode(self, z):
        x = self.decoder(z)
        return Bernoulli(logits=x)

    def sample(self, num_samples=64):
        z = self.prior.sample((num_samples,))
        sample = self.decode(z).probs
        return sample.view(num_samples, 1, 28, 28)

    def elbo(self, true_x, z, x_dist, z_dist):
        true_x = self.proc_data(true_x)
        lpxz = x_dist.log_prob(true_x).sum(-1) # equivalent to binary cross entropy.

        if self.analytic_kl and is_vae:
            # SGVB^B: -KL(q(z|x)||p(z)) + log p(x|z). Use when KL can be done analytically.
            assert z.size(0) == 1 and z.size(1) == 1
            kl = torch.distributions.kl.kl_divergence(z_dist, self.prior).sum(-1)
        else:
            # SGVB^A: log p(z) - log q(z|x) + log p(x|z)
            lpz = self.prior.log_prob(z).sum(-1)
            lqzx = z_dist.log_prob(z).sum(-1)
            kl = -lpz + lqzx
        return -kl + lpxz

    def logmeanexp(self, inputs, dim=1):
        if inputs.size(dim) == 1:
            return inputs
        else:
            input_max = inputs.max(dim, keepdim=True)[0]
            return (inputs - input_max).exp().mean(dim).log() + input_max

    def forward(self, true_x, mean_n, imp_n):
        z_dist = self.encode(true_x)
        z = z_dist.rsample(torch.Size([mean_n, imp_n])) # mean_n, imp_n, batch_size, z_dim
        x_dist = self.decode(z)

        elbo = self.elbo(true_x, z, x_dist, z_dist) # mean_n, imp_n, batch_size
        elbo_iwae = self.logmeanexp(elbo, 1).squeeze(1) # mean_n, batch_size
        elbo_iwae_m = torch.mean(elbo_iwae, 0) # batch_size
        return {'elbo': elbo, 'loss': -elbo_iwae_m}

