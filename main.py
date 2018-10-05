import os
import argparse
import torch
from torch import optim
from tensorboardX import SummaryWriter

from data_loader import data_loaders
from draw_figs import draw_figs
from vae import Vae

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--eval', type=bool, default=False)

parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128) # iwae uses 20
parser.add_argument('--epochs', type=int, default=3280)

parser.add_argument('--h_dim', type=int, default=200)
parser.add_argument('--z_dim', type=int, default=50)
parser.add_argument('--beta', type=float, default=1)
args = parser.parse_args()
if args.dataset == 'mnist':
    args.x_dim = 784
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
train_loader, test_loader = data_loaders(args)
torch.manual_seed(args.seed)
if args.cuda: torch.cuda.manual_seed_all(args.seed)
args.exp_name = 'h{}_z{}_lr{}_beta{}'.format(args.h_dim, args.z_dim, args.learning_rate, args.beta)
args.figs_dir = 'figs/{}'.format(args.exp_name)
args.out_dir = 'result/{}'.format(args.exp_name)
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
writer = SummaryWriter(args.out_dir)
if not os.path.exists(args.figs_dir):
    os.makedirs(args.figs_dir)

def train(epoch):
    global train_step
    for batch_idx, (data, _) in enumerate(train_loader):
        train_step += 1
        optimizer.zero_grad()
        outs = model(data)
        loss, elbo = model.loss(true_x=data, z=outs['z'], x_dist=outs['x_dist'], z_dist=outs['z_dist'])
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
        loss_sum += loss.item() * len(data)
        elbo_sum += elbo.item() * len(data)
    return loss_sum / len(test_loader.dataset), elbo_sum / len(test_loader.dataset)

model = VAE(device, x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim, beta=args.beta).to(device)
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
        if epoch % 10 == 1:
            draw_figs(model, args, test_loader, epoch)
            plt.close('all')
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

