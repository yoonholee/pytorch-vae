import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import pathlib
import numpy as np
import torch
from torch import optim
from torchvision.utils import save_image
from tensorboardX import SummaryWriter

import data_loader
from vae import VAE

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--eval', type=bool, default=False)

parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--h_dim', type=int, default=200)
parser.add_argument('--z_dim', type=int, default=50)
parser.add_argument('--beta', type=float, default=1)
args = parser.parse_args()
if args.dataset == 'mnist':
    args.x_dim = 784
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
train_loader, test_loader = data_loader.data_loaders(args)
torch.manual_seed(args.seed)
if args.cuda: torch.cuda.manual_seed_all(args.seed)
args.exp_name = 'h{}_z{}_lr{}_beta{}'.format(args.h_dim, args.z_dim, args.learning_rate, args.beta)
figs_dir = 'figs/{}'.format(args.exp_name)
out_dir = 'result/{}'.format(args.exp_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
writer = SummaryWriter(out_dir)
if not os.path.exists(figs_dir):
    os.makedirs(figs_dir)

def train(epoch):
    global train_step
    for batch_idx, (data, _) in enumerate(train_loader):
        train_step += 1
        data = data.to(device).view(-1, args.x_dim)
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
        data = data.to(device).view(-1, args.x_dim)
        outs = model(data)
        loss, elbo = model.loss(true_x=data, z=outs['z'], x_dist=outs['x_dist'], z_dist=outs['z_dist'])
        loss_sum += loss.item() * len(data)
        elbo_sum += elbo.item() * len(data)
    return loss_sum / len(test_loader.dataset), elbo_sum / len(test_loader.dataset)

def draw_figs(epoch):
    # TODO: ugly. refactor. can we move this to a separate file?
    samples = model.sample(num_samples=100).view(-1, 28, 28).data.cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.suptitle('Samples, Epoch {}'.format(epoch), fontsize=20)
    plt.axis('square')
    plt.legend(frameon=True)
    for idx, im in enumerate(samples):
        plt.subplot(10, 10, idx+1)
        plt.imshow(im, cmap='Greys')
        plt.axis('off')
    plt.savefig('figs/{}/samples_{:04}.jpg'.format(args.exp_name, epoch))
    plt.clf()

    files = [file for file in pathlib.Path(figs_dir).glob('samples_*.jpg')]
    images = [imageio.imread(str(file)) for file in sorted(files)]
    imageio.mimsave('{}/samples.gif'.format(figs_dir), images)

    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.to(device).view(-1, args.x_dim)
        break
    outs = model(data)
    data = data.view(args.batch_size, 28, 28)
    recon = outs['x_dist'].probs.view(args.batch_size, 28, 28)
    plt.figure(figsize=(5, 5))
    plt.suptitle('Reconstruction, Epoch {}'.format(epoch), fontsize=20)
    plt.axis('square')
    plt.legend(frameon=True)
    for i in range(50):
        data_i = data[i].data.cpu().numpy()
        recon_i = recon[i].data.cpu().numpy()
        plt.subplot(10, 10, 2*i+1)
        plt.imshow(data_i, cmap='Greys')
        plt.axis('off')
        plt.subplot(10, 10, 2*i+2)
        plt.imshow(recon_i, cmap='Greys')
        plt.axis('off')
    plt.savefig('figs/{}/reconstruction_{:04}.jpg'.format(args.exp_name, epoch))
    plt.clf()

    files = [file for file in pathlib.Path(figs_dir).glob('reconstruction_*.jpg')]
    images = [imageio.imread(str(file)) for file in sorted(files)]
    imageio.mimsave('{}/reconstruction.gif'.format(figs_dir), images)

    if args.z_dim == 2:
        latent_space, labels = [], []
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device).view(-1, args.x_dim)
            latent_space.append(model.encode(data).loc.data.cpu().numpy())
            labels.append(label)
        latent_space, labels = np.concatenate(latent_space), np.concatenate(labels)
        plt.figure(figsize=(5, 5))
        for c in range(10):
            idx = (labels == c)
            plt.scatter(latent_space[idx, 0], latent_space[idx, 1],
                        c=matplotlib.cm.get_cmap('tab10')(c), marker=',', label=str(c), alpha=.7)
        plt.suptitle('Latent representation, Epoch {}'.format(epoch), fontsize=20)
        plt.axis('square')
        plt.legend(frameon=True)
        plt.savefig('figs/{}/latent_{:04}.jpg'.format(args.exp_name, epoch))
        plt.clf()

        files = [file for file in pathlib.Path(figs_dir).glob('latent*.jpg')]
        images = [imageio.imread(str(file)) for file in sorted(files)]
        imageio.mimsave('{}/latent.gif'.format(figs_dir), images)

model = VAE(device, x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim, beta=args.beta).to(device)
if args.eval:
    model.load_state_dict(torch.load(out_dir+'/best_model.pt'))
    raise NotImplementedError
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

train_step = 0
least_loss = float('inf')
for epoch in range(1, args.epochs + 1):
    train(epoch)
    with torch.no_grad():
        draw_figs(epoch)
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
            torch.save(model.state_dict(), out_dir+'/best_model.pt')

