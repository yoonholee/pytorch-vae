import os
import sys
import numpy as np
import torch
from torch import optim
from tensorboardX import SummaryWriter
from config import get_args
from data_loader.data_loader import data_loaders
from model.bernoulli_vae import BernoulliVAE
from model.conv_vae import ConvVAE
from to_sheets import upload_to_google_sheets
from draw_figs import draw_figs

args = get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
args.cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")
train_loader, test_loader = data_loaders(args)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed_all(args.seed)
writer = SummaryWriter(args.out_dir)


def train(epoch):
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        outs = model(data, mean_n=args.mean_num, imp_n=args.importance_num)
        loss_1, loss = -outs['elbo'].cpu().data.numpy().mean(), outs['loss'].mean()
        loss.backward()
        optimizer.step()

        model.train_step += 1
        if model.train_step % args.log_interval == 0:
            print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()))
            writer.add_scalar('train/loss', loss.item(), model.train_step)
            writer.add_scalar('train/loss_1', loss_1, model.train_step)


def test(epoch):
    elbos = [model(data, mean_n=1, imp_n=5000)['elbo'].squeeze(0) for data, _ in test_loader]

    def get_loss_k(k):
        losses = [model.logmeanexp(elbo[:k], 0).cpu().numpy().flatten() for elbo in elbos]
        return -np.concatenate(losses).mean()
    return map(get_loss_k, [args.importance_num, 1, 64, 5000])


model_class = BernoulliVAE if args.arch == 'bernoulli' else ConvVAE
mean_img = train_loader.dataset.get_mean_img()
model = model_class(device, x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim,
                    analytic_kl=args.analytic_kl, mean_img=mean_img).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-4)
if args.no_iwae_lr:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=100, factor=10**(-1/7))
else:
    milestones = np.cumsum([3**i for i in range(8)])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=10**(-1/7))

if args.eval:
    model.load_state_dict(torch.load(args.best_model_file))
    with torch.no_grad():
        print(list(test(0)))
        if args.figs:
            draw_figs(model, args, test_loader, 0)
    sys.exit()

for epoch in range(1, args.epochs):
    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
    train(epoch)
    with torch.no_grad():
        if args.figs and epoch % 100 == 1:
            draw_figs(model, args, test_loader, epoch)
        test_loss, test_1, test_64, test_5000 = test(epoch)
        if test_loss < model.best_loss:
            model.best_loss = test_loss
            torch.save(model.state_dict(), args.best_model_file)
        scheduler_args = {'metrics': test_loss} if args.no_iwae_lr else {}
        scheduler.step(**scheduler_args)
        writer.add_scalar('test/loss', test_loss, epoch)
        writer.add_scalar('test/loss_1', test_1, epoch)
        writer.add_scalar('test/loss_64', test_64, epoch)
        writer.add_scalar('test/LL', test_5000, epoch)
        print('==== Testing. LL: {:.4f} ====\n'.format(test_5000))

row_data = [args.exp_name, str(test_5000), str(test_64), str(test_64-test_5000)]
upload_to_google_sheets(row_data=row_data)
