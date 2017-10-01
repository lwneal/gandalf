import sys
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
from gradient_penalty import calc_gradient_penalty
from torch.nn.functional import nll_loss
from sklearn.decomposition import PCA
import imutil


def to_np(v):
    return v.data.cpu().numpy()


def evaluate_classifier(networks, dataloader, **options):
    netE = networks['encoder']
    netG = networks['generator']
    netC = networks['classifier']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']

    correct = 0
    total = 0
    mae = 0
    mse = 0
    latent_vectors = []
    plot_labels = []
    
    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images, volatile=True)
        z = netE(images)
        class_predictions = netC(z)

        # https://discuss.pytorch.org/t/argmax-with-pytorch/1528/2
        _, predicted = class_predictions.max(1)
        correct += sum(predicted.data == labels)
        total += len(predicted)

        latent_vectors.extend(z.data.cpu().numpy())
        plot_labels.extend(labels.cpu().numpy())

        reconstructed = netG(z)
        mae += torch.mean(torch.abs(reconstructed - images))
        mse += torch.mean((reconstructed - images) ** 2)
        print("Accuracy: {:.4f} ({: >12} / {: <12} correct)".format(float(correct) / total, correct, total))

    # Save latent vectors for later visualization
    latent_vectors = np.array(latent_vectors)
    filename = 'z_{}_epoch_{:04d}.npy'.format(options['fold'], options['epoch'])
    filename = os.path.join(result_dir, filename)
    if options.get('save_latent_vectors'):
        np.save(filename, latent_vectors)

    # Run PCA on the latent vectors to generate a 2d visualization
    pca_vectors = pca(latent_vectors)
    plot_filename = 'plot_pca_{}_epoch{:04d}.png'.format(options['fold'], options['epoch'])
    plot_filename = os.path.join(result_dir, plot_filename)
    title = 'PCA: {} epoch {}'.format(options['fold'], options['epoch'])
    plot(pca_vectors, plot_filename, title=title, labels=plot_labels)

    mse = float(to_np(mse / i)[0])
    mae = float(to_np(mae / i)[0])
    print("Reconstruction per-pixel MSE: {}".format(mse))
    print("Reconstruction per-pixel MAE: {}".format(mae))

    return {
        options['fold']: {
            'correct': correct,
            'total': total,
            'mse': mse,
            'mae': mae,
            'accuracy': float(correct) / total,
            'latent_vectors': filename,
        }
    }


def pca(vectors):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(vectors)
    return pca.transform(vectors)


# Plots a list of 2d points
def plot(dots, output_filename, title=None, labels=None):
    # Incantation to enable headless mode
    import matplotlib
    matplotlib.use('Agg')
    # Apply sane Seaborn defaults to Matplotlib
    import seaborn as sns
    sns.set_style('darkgrid')
    import matplotlib.pyplot as plt

    import pandas as pd
    df = pd.DataFrame(dots)
    df.columns = ['Z_1', 'Z_2']
    df['label'] = labels
    plot = sns.pairplot(df, size=4, hue='label', plot_kws={'s':12})
    if title:
        plot.fig.suptitle(title)
    plot.savefig(output_filename)
