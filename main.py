import os
import argparse
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
        train_step += 1
        optimizer.zero_grad()
        if args.importance_num == 1: # vae
            outs = model(data)
            loss, elbo = model.loss(true_x=data, z=outs['z'], x_dist=outs['x_dist'], z_dist=outs['z_dist'])
        else:
            elbos = []
            for _ in range(args.importance_num):
                outs = model(data)
                _, elbo = model.loss(true_x=data, z=outs['z'], x_dist=outs['x_dist'], z_dist=outs['z_dist'])
                elbos.append(elbo)
            elbo = model.iwae_bound(elbos)
            loss = -elbo
        loss, elbo = loss.mean(), elbo.mean()
        writer.add_scalar('train/loss', loss.item(), train_step)
        writer.add_scalar('train/elbo', elbo.item(), train_step)
        loss.backward()
        optimizer.step()
        if train_step % args.log_interval == 0:
            print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()))

def test(epoch, k):
    elbo_sum = 0
    for batch_idx, (data, _) in enumerate(test_loader):
        elbos = []
        for _ in range(k):
            outs = model(data)
            _, elbo = model.loss(true_x=data, z=outs['z'], x_dist=outs['x_dist'], z_dist=outs['z_dist'])
            elbos.append(elbo)
            import pdb; pdb.set_trace() # TODO: make this scale to k = 5000
        elbo = model.iwae_bound(elbos)
        elbo_sum += elbo.sum()
    return elbo_sum / len(test_loader.dataset)

model = VAE(device, x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim,
            beta=args.beta, analytic_kl=args.analytic_kl).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

train_step = 0
learning_rates = [1e-3 * 10**-(j/7) for j in sum([[i] * 3**i for i in range(8)], [])]
for epoch in range(1, args.epochs + 1):
    optimizer.param_groups[0]['lr'] = learning_rates[epoch - 1]
    train(epoch)
    import time
    for i in range(100):
        start = time.time()
        print(2**i, test(epoch, 2**i))
        print(time.time() - start)
    with torch.no_grad():
        if args.figs and epoch % 10 == 1:
            draw_figs(model, args, test_loader, epoch)
        test_elbo = test(epoch)
        print('==== elbo: {:.4f} current lr: {} ====\n'.format(
            test_elbo, optimizer.param_groups[0]['lr']))
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('test/elbo', test_elbo, epoch)

