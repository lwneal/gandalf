import json
import sys
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
from gradient_penalty import calc_gradient_penalty
from torch.nn.functional import nll_loss
from torch.nn.functional import softmax, sigmoid
import imutil
from sklearn.metrics import roc_curve, auc


def to_np(v):
    return v.data.cpu().numpy()


def evaluate_classifier(networks, dataloader, verbose=True, skip_reconstruction=False, **options):
    for net in networks.values():
        net.eval()
    netE = networks['encoder']
    netG = networks['generator']
    netC = networks['classifier']
    netD = networks['discriminator']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']

    classification_correct = 0
    classification_total = 0
    openset_correct = 0
    openset_total = 0
    combined_correct = 0
    combined_total = 0

    # TODO: Hard-coded for MNIST/SVHN open set
    num_classes = 6
    openset_preds = []
    openset_labels = []
    
    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images, volatile=True)
        z = netE(images)

        # Predict a classification among known classes
        net_y = netC(z)[:, :num_classes]
        class_predictions = softmax(net_y)

        _, predicted = class_predictions.max(1)
        classification_correct += sum(predicted.data == labels)
        classification_total += sum(labels < num_classes)

        max_vals, max_idx = net_y.max(dim=1)
        pred_openset = -max_vals.data.cpu().numpy()
        label_openset = (labels >= num_classes).cpu().numpy()

        openset_preds.extend(pred_openset)
        openset_labels.extend(label_openset)

        combined_correct += ((labels >= num_classes) * (max_vals.data < 0) + (predicted.data == labels)).sum()
        combined_total += len(images)

    fpr, tpr, thresholds = roc_curve(openset_labels, openset_preds)
    openset_auc = auc(fpr, tpr)

    stats = {
        options['fold']: {
            'correct': classification_correct,
            'total': classification_total,
            'accuracy': float(classification_correct) / classification_total,
            'classification_accuracy': float(classification_correct) / classification_total,
            'combined_accuracy': float(combined_correct) / combined_total,
            'openset_auc': openset_auc,
        }
    }
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
    for net in networks.values():
        net.eval()
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
