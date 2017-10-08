import json
import sys
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
from gradient_penalty import calc_gradient_penalty
from torch.nn.functional import nll_loss
import imutil


def to_np(v):
    return v.data.cpu().numpy()


def evaluate_classifier(networks, dataloader, verbose=True, **options):
    netE = networks['encoder']
    netG = networks['generator']
    netC = networks['classifier']
    netD = networks['discriminator']
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
    discriminator_scores = []
    
    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images, volatile=True)
        z = netE(images)

        reconstructed = netG(z)
        mae += torch.mean(torch.abs(reconstructed - images))
        mse += torch.mean((reconstructed - images) ** 2)

        class_predictions = netC(z)
        _, predicted = class_predictions.max(1)
        correct += sum(predicted.data == labels)
        total += len(predicted)

        latent_vectors.extend(z.data.cpu().numpy())
        plot_labels.extend(labels.cpu().numpy())

        discriminator_scores.extend(netD(images).data.cpu().numpy())

        if verbose:
            print("Accuracy: {:.4f} ({: >12} / {: <12} correct)".format(float(correct) / total, correct, total))


    # Save latent vectors for later visualization
    latent_vectors = np.array(latent_vectors)
    if options.get('save_latent_vectors'):
        z_filename = 'z_{}_epoch_{:04d}.npy'.format(options['fold'], options['epoch'])
        z_filename = os.path.join(result_dir, z_filename)
        np.save(z_filename, latent_vectors)

    if verbose:
        # Run PCA on the latent vectors to generate a 2d visualization
        pca_vectors = pca(latent_vectors)
        plot_filename = 'plot_pca_{}_epoch_{:04d}.png'.format(options['fold'], options['epoch'])
        plot_filename = os.path.join(result_dir, plot_filename)
        title = 'PCA: {} epoch {}'.format(options['fold'], options['epoch'])
        plot(pca_vectors, plot_filename, title=title, labels=plot_labels)

        # Also visualize the first two dimensions of the latent space
        latent_2d = latent_vectors[:, :2]
        plot_filename = 'plot_latent2_{}_epoch_{:04d}.png'.format(options['fold'], options['epoch'])
        plot_filename = os.path.join(result_dir, plot_filename)
        title = 'Latent Dim #0 vs #1: {} epoch {}'.format(options['fold'], options['epoch'])
        plot(latent_2d, plot_filename, title=title, labels=plot_labels)

    mse = float(to_np(mse / i)[0])
    mae = float(to_np(mae / i)[0])
    print("Reconstruction per-pixel MSE: {}".format(mse))
    print("Reconstruction per-pixel MAE: {}".format(mae))

    discriminator_mean = float(np.array(discriminator_scores).mean())

    stats = {
        options['fold']: {
            'correct': correct,
            'total': total,
            'mse': mse,
            'mae': mae,
            'accuracy': float(correct) / total,
            'discriminator_mean': discriminator_mean,
        }
    }
    if options.get('save_latent_vectors'):
        stats[options['fold']]['latent_vectors'] = z_filename
    return stats


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


# Open Set Classification
# Given two datasets, one on-manifold and another off-manifold, predict
# whether each item is on-manifold or off-manifold using the discriminator
# or the autoencoder loss.
# Plot an ROC curve for each and report AUC
# dataloader_on: Test set of the same items the network was trained on
# dataloader_off: Separate dataset from a different distribution
def evaluate_openset(networks, dataloader_on, dataloader_off, **options):
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']

    mae_scores_on, mse_scores_on, d_scores_on, c_scores_on = get_openset_scores(dataloader_on, networks)
    mae_scores_off, mse_scores_off, d_scores_off, c_scores_off = get_openset_scores(dataloader_off, networks)

    y_true = np.array([0] * len(d_scores_on) + [1] * len(d_scores_off))
    y_discriminator = np.concatenate([d_scores_on, d_scores_off])
    y_mae = np.concatenate([mae_scores_on, mae_scores_off])
    y_mse = np.concatenate([mse_scores_on, mse_scores_off])
    y_softmax = np.concatenate([c_scores_on, c_scores_off])

    y_combined2 = combine_scores([y_discriminator, y_mae])
    y_combined3 = combine_scores([y_discriminator, y_mae, y_softmax])

    auc_d, plot_d = plot_roc(y_true, y_discriminator, 'Discriminator ROC vs {}'.format(dataloader_off.dsf.name))
    auc_mae, plot_mae = plot_roc(y_true, y_mae, 'Reconstruction MAE ROC vs {}'.format(dataloader_off.dsf.name))
    auc_mse, plot_mse = plot_roc(y_true, y_mse, 'Reconstruction MSE ROC vs {}'.format(dataloader_off.dsf.name))
    auc_softmax, plot_softmax = plot_roc(y_true, y_softmax, 'Softmax ROC vs {}'.format(dataloader_off.dsf.name))
    auc_combined2, plot_combined2 = plot_roc(y_true, y_combined2, 'Combined-2 ROC vs {}'.format(dataloader_off.dsf.name))
    auc_combined3, plot_combined3 = plot_roc(y_true, y_combined3, 'Combined-3 ROC vs {}'.format(dataloader_off.dsf.name))

    save_plot(plot_d, 'roc_discriminator', **options)
    save_plot(plot_mae, 'roc_mae', **options)
    save_plot(plot_mse, 'roc_mse', **options)
    save_plot(plot_softmax, 'roc_softmax', **options)
    save_plot(plot_combined2, 'roc_combined2', **options)
    save_plot(plot_combined3, 'roc_combined3', **options)

    return {
        'auc_discriminator': auc_d,
        'auc_mae': auc_mae,
        'auc_mse': auc_mse,
        'auc_softmax': auc_softmax,
        'auc_combined2': auc_combined2,
        'auc_combined3': auc_combined3,
    }


def combine_scores(score_list):
    example_count = len(score_list[0])
    assert all(len(x) == example_count for x in score_list)

    normalized_scores = np.ones(example_count)
    for score in score_list:
        score -= score.min()
        score /= score.max()
        normalized_scores *= score
        normalized_scores /= normalized_scores.max()
    return normalized_scores


def save_plot(plot, title, **options):
    comparison_name = options['comparison_dataset'].split('/')[-1].replace('.dataset', '')
    filename = 'plot_{}_vs_{}_epoch_{:04d}.png'.format(title, comparison_name, options['epoch'])
    filename = os.path.join(options['result_dir'], filename)
    plot.figure.savefig(filename)
    

def get_openset_scores(dataloader, networks):
    netE = networks['encoder']
    netG = networks['generator']
    netD = networks['discriminator']
    netC = networks['classifier']

    mae_scores = []
    mse_scores = []
    discriminator_scores = []
    softmax_scores = []

    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images, volatile=True)
        z = netE(images)
        reconstructed = netG(z)

        # Classification via autoencoder reconstruction error
        mae = torch.abs(reconstructed - images)
        mae = mae.mean(1).mean(1).mean(1)
        mae_scores.extend(mae.data.cpu().numpy())

        mse = (reconstructed - images) ** 2
        mse = mse.mean(1).mean(1).mean(1)
        mse_scores.extend(mse.data.cpu().numpy())

        # Classification directly via the discriminator
        discriminator_scores.extend(netD(images).data.cpu().numpy())

        classifier_scores = netC(z)
        softmax = -torch.exp(classifier_scores.max(1)[0])
        softmax_scores.extend(softmax.data.cpu().numpy())

    mae_scores = np.array(mae_scores)
    mse_scores = np.array(mse_scores)
    discriminator_scores = np.array(discriminator_scores)
    softmax_scores = np.array(softmax_scores)
    return mae_scores, mse_scores, discriminator_scores, softmax_scores


def plot_roc(y_true, y_score, title="Receiver Operating Characteristic"):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    from plotting import plot_xy
    plot = plot_xy(fpr, tpr, x_axis="False Positive Rate", y_axis="True Positive Rate", title=title)
    return auc_score, plot


def save_evaluation(new_results, result_dir, epoch):
    filename = 'eval_epoch_{:04d}.json'.format(epoch)
    filename = os.path.join(result_dir, filename)
    filename = os.path.expanduser(filename)
    if os.path.exists(filename):
        old_results = json.load(open(filename))
    else:
        old_results = {}
    old_results.update(new_results)
    with open(filename, 'w') as fp:
        json.dump(old_results, fp, indent=2, sort_keys=True)
