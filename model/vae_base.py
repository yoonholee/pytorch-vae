import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal


class VAE(nn.Module):
    def __init__(self, device, z_dim, analytic_kl):
        super().__init__()
        self.train_step = 0
        self.best_loss = np.inf
        self.analytic_kl = analytic_kl
        self.prior = Normal(
            torch.zeros([z_dim]).to(device), torch.ones([z_dim]).to(device)
        )

    def proc_data(self, x):
        pass

    def encode(self, x):
        pass

    def decode(self, z):
        pass

    def lpxz(self, true_x, x_dist):
        pass

    def sample(self, num_samples=64):
        pass

    def elbo(self, true_x, z, x_dist, z_dist):
        true_x = self.proc_data(true_x)
        lpxz = self.lpxz(true_x, x_dist)

        if self.analytic_kl:
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
        # mean_n, imp_n, batch_size, z_dim
        z = z_dist.rsample(torch.Size([mean_n, imp_n]))
        x_dist = self.decode(z)

        elbo = self.elbo(true_x, z, x_dist, z_dist)  # mean_n, imp_n, batch_size
        elbo_iwae = self.logmeanexp(elbo, 1).squeeze(1)  # mean_n, batch_size
        elbo_iwae_m = torch.mean(elbo_iwae, 0)  # batch_size
        return {"elbo": elbo, "loss": -elbo_iwae_m}
