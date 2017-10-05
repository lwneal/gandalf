#!/usr/bin/env python
import numpy as np
import sys
import requests
import os
import json
from PIL import Image
from tqdm import tqdm

DATA_DIR = os.path.expanduser('~/data')

def save_set(fold, x, y, suffix='png'):
    examples = []
    fp = open('mnist_{}.dataset'.format(fold), 'w')
    print("Writing MNIST dataset {}".format(fold))
    for i in tqdm(range(len(x))):
        label = y[i]
        img_filename = '~/data/mnist/{}/{:05d}_{:d}.{}'.format(fold, i, label, suffix)
        if not os.path.exists(img_filename):
            Image.fromarray(x[i]).save(os.path.expanduser(img_filename))
        entry = {
                'filename': img_filename,
                'label': str(label),
                'is_holy': label in [0, 6, 8, 9],  # numbers with holes in them
                'is_pointy': label in [1, 4, 7],  # numbers with no curvy parts
                'is_symmetric': label in [0, 1, 8],  # left-right symmetric numbers
                'label': str(label),
                'fold': fold,
        }
        examples.append(entry)
        fp.write(json.dumps(entry))
        fp.write('\n')
    fp.close()
    return examples


def download_mnist_data(path='mnist.npz'):
    if not os.path.exists(path):
        response = requests.get('https://s3.amazonaws.com/img-datasets/mnist.npz')
        open(path, 'wb').write(response.content)
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def mkdir(dirname):
    import errno
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        if not os.path.isdir(dirname):
            raise


if __name__ == '__main__':
    os.chdir(DATA_DIR)
    mkdir('mnist')
    mkdir('mnist/train')
    mkdir('mnist/test')
    (train_x, train_y), (test_x, test_y) = download_mnist_data()

    train = save_set('train', train_x, train_y)
    test = save_set('test', test_x, test_y)
    for example in train:
        example['fold'] = 'train'
    for example in test:
        example['fold'] = 'test'
    with open('mnist.dataset', 'w') as fp:
        for example in train + test:
            fp.write(json.dumps(example) + '\n')

    with open('mnist-05.dataset', 'w') as fp:
        for example in train + test:
            if int(example['label']) < 6:
                fp.write(json.dumps(example) + '\n')
    with open('mnist-69.dataset', 'w') as fp:
        for example in train + test:
            if int(example['label']) >= 6:
                fp.write(json.dumps(example) + '\n')
