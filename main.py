import os
import argparse
import scipy.special
import numpy as np
import torch
from torch import optim
from tensorboardX import SummaryWriter

from config import get_args
from data_loader import data_loaders
from vae import VAE

args = get_args()
if args.figs:
    from draw_figs import draw_figs
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
train_loader, test_loader = data_loaders(args)
torch.manual_seed(args.seed)
if args.cuda: torch.cuda.manual_seed_all(args.seed)
writer = SummaryWriter(args.out_dir)

def loss_from_elbos(elbos, n):
    if n == -1: n = len(elbos)
    return elbos.mean() if n == 1 else \
            scipy.special.logsumexp(elbos[:n], 0) - np.log(n)

def train(epoch):
    global train_step
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        outs = model(data, mean_n=args.mean_num, imp_n=args.importance_num)
        elbo, loss = outs['elbo'].cpu().data.numpy(), outs['loss'].mean()

        train_step += 1
        loss.backward()
        optimizer.step()
        if train_step % args.log_interval == 0:
            print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalar('train/loss', loss.item(), train_step)
            writer.add_scalar('train/loss_1', loss_from_elbos(elbo, 1), train_step)

def test(epoch):
    elbo = [model(data, mean_n=1, imp_n=5000)['elbo'].cpu().data.numpy()
            for data, _ in test_loader]
    elbo = np.concatenate(elbo, -1).squeeze(0)
    print('==== Testing. LL: {:.4f} current lr: {} ====\n'.format(
        loss_from_elbos(elbo, -1).mean(), optimizer.param_groups[0]['lr']))
    writer.add_scalar('test/loss_1', loss_from_elbos(elbo, 1).mean(), epoch)
    writer.add_scalar('test/loss_64', loss_from_elbos(elbo, 64).mean(), epoch)
    writer.add_scalar('test/LL', loss_from_elbos(elbo, -1).mean(), epoch)

model = VAE(device, x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim,
            beta=args.beta, analytic_kl=args.analytic_kl).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

train_step = 0
learning_rates = [1e-3 * 10**-(j/7) for j in sum([[i] * 3**i for i in range(8)], [])]
for epoch in range(1, args.epochs + 1):
    optimizer.param_groups[0]['lr'] = learning_rates[epoch - 1]
    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
    train(epoch)
    with torch.no_grad():
        if epoch % 10 == 1: test(epoch)
        if args.figs and epoch % 10 == 1: draw_figs(model, args, test_loader, epoch)

