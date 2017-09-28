import sys
import time
import torch
from torch.autograd import Variable
from gradient_penalty import calc_gradient_penalty
from torch.nn.functional import nll_loss
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

    latent_points = []
    correct = 0
    total = 0
    mae = 0
    mse = 0
    
    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images)
        class_predictions = netC(netE(images))

        # https://discuss.pytorch.org/t/argmax-with-pytorch/1528/2
        _, predicted = class_predictions.max(1)
        correct += sum(predicted.data == labels)
        total += len(predicted)

        z = netE(images)
        latent_points.extend(z)
        reconstructed = netG(z)
        mae += torch.mean(torch.abs(reconstructed - images))
        mse += torch.mean((reconstructed - images) ** 2)
        print("Accuracy: {:.4f} ({: >12} / {: <12} correct)".format(float(correct) / total, correct, total))

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
        }
    }
