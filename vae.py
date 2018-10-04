import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal

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

    def encode(self, x):
        h1 = F.relu(self.enc_fc1(x))
        mu, _std = self.enc_fc2_mu(h1), self.enc_fc2_sig(h1)
        std = F.softplus(_std)  # some implementations use exp(_std/2)
        z_dist = Normal(mu, std)
        return mu, z_dist

    def decode(self, z):
        h3 = F.relu(self.dec_fc1(z))
        recon_x = self.dec_fc2(h3)
        return recon_x

    def forward(self, x):
        mu, z_dist = self.encode(x.view(-1, self.x_dim))
        z = z_dist.rsample() if self.training else mu
        recon_x = self.decode(z)

        loss, var_bound = self.loss_function(x, recon_x, z_dist)
        return {'recon_x': recon_x, 'loss': loss, 'var_bound': var_bound}

    def loss_function(self, true_x, recon_x, z_dist):
        # SGVB(1): loss = log p(x,z) - log q(z|x)

        # KL can be done analytically: loss = -KL(q(z|x)||p(z)) + log p(x|z)
        BCE = F.binary_cross_entropy_with_logits(
            recon_x, true_x.view(-1, self.x_dim), reduction='none').sum(1).mean()
        KLD = torch.distributions.kl.kl_divergence(z_dist, self.prior).sum(1).mean()
        loss = BCE + self.beta * KLD
        var_bound = BCE + KLD
        return loss, var_bound

    def sample(self, num_samples=64):
        z = self.prior.sample((num_samples,))
        sample = self.decode(z)
        return sample.view(num_samples, 1, 28, 28)

