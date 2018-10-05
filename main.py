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
args.cuda = torch.cuda.is_available()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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
            for k in range(args.importance_num):
                outs = model(data)
                _, elbo = model.loss(true_x=data, z=outs['z'], x_dist=outs['x_dist'], z_dist=outs['z_dist'])
                elbos.append(elbo)
            loss = model.iwae_loss(elbos)
            elbo = torch.stack(elbos)
        loss, elbo = loss.mean(), elbo.mean()
        loss.backward()
        writer.add_scalar('train/loss', loss.item(), train_step)
        writer.add_scalar('train/elbo', elbo.item(), train_step)
        optimizer.step()
        if train_step % args.log_interval == 0:
            print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()))

def test(epoch):
    loss_sum, elbo_sum = 0, 0
    for batch_idx, (data, _) in enumerate(test_loader):
        outs = model(data)
        loss, elbo = model.loss(true_x=data, z=outs['z'], x_dist=outs['x_dist'], z_dist=outs['z_dist'])
        loss, elbo = loss.mean(), elbo.mean()
        loss_sum += loss.item() * len(data)
        elbo_sum += elbo.item() * len(data)
    return loss_sum / len(test_loader.dataset), elbo_sum / len(test_loader.dataset)

model = VAE(device, x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim,
            beta=args.beta, analytic_kl=args.analytic_kl).to(device)
if args.eval:
    model.load_state_dict(torch.load(args.out_dir+'/best_model.pt'))
    raise NotImplementedError
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

train_step = 0
least_loss = float('inf')
for epoch in range(1, args.epochs + 1):
    train(epoch)
    with torch.no_grad():
        if args.figs and epoch % 10 == 1:
            draw_figs(model, args, test_loader, epoch)
        test_loss, test_elbo = test(epoch)
        scheduler.step(test_loss)
        print('==== Test loss: {:.4f} elbo: {:.4f} current lr: {} ====\n'.format(
            test_loss, test_elbo, optimizer.param_groups[0]['lr']))
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], train_step)
        writer.add_scalar('test/loss', test_loss, train_step)
        writer.add_scalar('test/elbo', test_elbo, train_step)
        if test_loss < least_loss:  # early stopping
            least_loss = test_loss
            torch.save(model.state_dict(), args.out_dir+'/best_model.pt')

