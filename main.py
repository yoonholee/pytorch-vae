import os
import argparse
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

def train(epoch):
    global train_step
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        outs = model(data, mean_n=args.mean_num, imp_n=args.importance_num)
        loss_1, loss = -outs['elbo'].cpu().data.numpy().mean(), outs['loss'].mean()

        train_step += 1
        loss.backward()
        optimizer.step()
        if train_step % args.log_interval == 0:
            print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalar('train/loss', loss.item(), train_step)
            writer.add_scalar('train/loss_1', loss_1, train_step)

def test(epoch):
    elbos = [model(data, mean_n=1, imp_n=5000)['elbo'].squeeze(0) for data, _ in test_loader]

    def get_loss_k(elbos, k):
        losses = [model.logmeanexp(elbo[:k], 0).cpu().numpy().flatten() for elbo in elbos]
        return -np.concatenate(losses).mean()
    #TODO: get_loss_k calls redundant here. clean.
    print('==== Testing. LL: {:.4f} current lr: {} ====\n'.format(
        get_loss_k(elbos, -1), optimizer.param_groups[0]['lr']))
    writer.add_scalar('test/loss', get_loss_k(elbos, args.importance_num), epoch)
    writer.add_scalar('test/loss_1', get_loss_k(elbos, 1), epoch)
    writer.add_scalar('test/loss_64', get_loss_k(elbos, 64), epoch)
    writer.add_scalar('test/LL', get_loss_k(elbos, -1), epoch)
    return get_loss_k(elbos, args.importance_num)

mean_img = (train_loader.dataset.train_data.type(torch.float) / 255).mean(0).reshape(-1).numpy()
model = VAE(device, x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim,
            beta=args.beta, analytic_kl=args.analytic_kl, mean_img=mean_img).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-4)

train_step = 0
if args.iwae_lr:
    learning_rates = [1e-3 * 10**-(j/7) for j in sum([[i] * 3**i for i in range(8)], [])]
    for epoch in range(1, len(learning_rates)+1):
        optimizer.param_groups[0]['lr'] = learning_rates[epoch - 1]
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        train(epoch)
        with torch.no_grad():
            test_loss = test(epoch)
            if args.figs and epoch % 10 == 1: draw_figs(model, args, test_loader, epoch)
else:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', cooldown=20, factor=10**(-1/7))
    for epoch in range(1, args.epochs):
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        train(epoch)
        with torch.no_grad():
            test_loss = test(epoch)
            scheduler.step(test_loss)
            if args.figs and epoch % 10 == 1: draw_figs(model, args, test_loader, epoch)

