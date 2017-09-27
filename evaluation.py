import time
import torch
from torch.autograd import Variable
from gradient_penalty import calc_gradient_penalty
from torch.nn.functional import nll_loss
import imutil


def evaluate_classifier(networks, dataloader, **options):
    netE = networks['encoder']
    netC = networks['classifier']
    result_dir = options['result_dir']
    batch_size = options['batch_size']
    image_size = options['image_size']
    latent_size = options['latent_size']

    correct = 0
    total = 0
    
    for i, (images, labels) in enumerate(dataloader):
        images = Variable(images)
        class_predictions = netC(netE(images))

        # https://discuss.pytorch.org/t/argmax-with-pytorch/1528/2
        _, predicted = class_predictions.max(1)
        correct += sum(predicted.data == labels)
        total += len(predicted)

        print("Accuracy: {:.4f} ({: >12} / {: <12} correct)".format(float(correct) / total, correct, total))

    return {
        options['fold']: {
            'correct': correct,
            'total': total,
            'accuracy': float(correct) / total,
        }
    }
