import os
import argparse
import torch
from torch import optim
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import data_loader
from vae import VAE

parser = argparse.ArgumentParser(description='VAE MNIST')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--eval', type=bool, default=False)

parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=50)

parser.add_argument('--h_dim', type=int, default=400)
parser.add_argument('--z_dim', type=int, default=20)
parser.add_argument('--beta', type=float, default=1)
args = parser.parse_args()
if args.dataset == 'mnist':
    args.x_dim = 784
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
train_loader, test_loader = data_loader.data_loaders(args)
torch.manual_seed(args.seed)
out_dir = 'result/h{}_z{}_lr{}_beta{}'.format(
    args.h_dim, args.z_dim, args.learning_rate, args.beta)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
writer = SummaryWriter(out_dir)

def train(epoch):
    global train_step
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        train_step += 1
        data = data.to(device)
        optimizer.zero_grad()
        outs = model(data)
        loss = outs['loss']
        loss.backward()
        writer.add_scalar('train/loss', loss.item(), train_step)
        optimizer.step()
        if train_step % args.log_interval == 0:
            print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()))

def test(epoch):
    global train_step
    model.eval()
    test_loss, test_var_bound = 0, 0
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.to(device)
        outs = model(data)
        test_loss += outs['loss'].item() * len(data)
        test_var_bound += outs['var_bound'].item() * len(data)
        if batch_idx == 0:
            comparison = torch.cat([data[:8], outs['recon_x'].view(args.batch_size, 1, 28, 28)[:8]])
            save_image(comparison, out_dir + '/reconstruction_' + str(epoch) + '.png', nrow=8)
    return test_loss / len(test_loader.dataset), test_var_bound / len(test_loader.dataset)

model = VAE(device, x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim,
            beta=args.beta).to(device)
if args.eval:
    model.load_state_dict(torch.load(out_dir+'/best_model.pt'))
    raise NotImplementedError
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
train_step = 0
least_loss = float('inf')
for epoch in range(1, args.epochs + 1):
    train(epoch)
    with torch.no_grad():
        sample = model.sample(num_samples=64)
        save_image(sample, out_dir + '/sample_' + str(epoch) + '.png')

        test_loss, test_var_bound = test(epoch)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        print('====> Test set variational bound: {:.4f}'.format(test_var_bound))
        writer.add_scalar('test/loss', test_loss, train_step)
        writer.add_scalar('test/var_bound', test_var_bound, train_step)
        if test_loss < least_loss:  # early stopping
            least_loss = test_loss
            torch.save(model.state_dict(), out_dir+'/best_model.pt')

