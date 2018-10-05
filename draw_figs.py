import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import pathlib
import numpy as np
import torch

def draw_figs(model, args, test_loader, epoch):
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

    files = [file for file in pathlib.Path(args.figs_dir).glob('samples_*.jpg')]
    images = [imageio.imread(str(file)) for file in sorted(files)]
    imageio.mimsave('{}/samples.gif'.format(args.figs_dir), images)

    for batch_idx, (data, _) in enumerate(test_loader):
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

    files = [file for file in pathlib.Path(args.figs_dir).glob('reconstruction_*.jpg')]
    images = [imageio.imread(str(file)) for file in sorted(files)]
    imageio.mimsave('{}/reconstruction.gif'.format(args.figs_dir), images)

    if args.z_dim == 2:
        latent_space, labels = [], []
        for batch_idx, (data, label) in enumerate(test_loader):
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

        files = [file for file in pathlib.Path(args.figs_dir).glob('latent*.jpg')]
        images = [imageio.imread(str(file)) for file in sorted(files)]
        imageio.mimsave('{}/latent.gif'.format(args.figs_dir), images)

