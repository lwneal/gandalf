"""
A Converter converts between:
    examples (each one a dict with keys like "filename" and "label")
    arrays (numpy arrays input to or output from a network)

Dataset augmentation can be accomplished with a Converter that returns a
different array each time to_array is called with the same example
"""
import os
import numpy as np
import random
import imutil


# Converters can be used like a function, on a single example or a batch
class Converter(object):
    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            return [self.from_array(e) for e in inputs]
        elif isinstance(inputs, list):
            return np.array([self.to_array(e) for e in inputs])
        else:
            return self.to_array(inputs)


# Crops, resizes, normalizes, performs any desired augmentations
# Outputs images as eg. 32x32x3 np.array or eg. 3x32x32 torch.FloatTensor
class ImageConverter(Converter):
    def __init__(self, 
            dataset,
            image_size=32,
            crop_to_bounding_box=True,
            random_horizontal_flip=False,
            delete_background=False,
            torch=True,
            normalize=True,
            **kwargs):
        width, height = image_size, image_size
        self.img_shape = (width, height)
        self.bounding_box = crop_to_bounding_box
        self.data_dir = dataset.data_dir
        self.random_horizontal_flip = random_horizontal_flip
        self.torch = torch
        self.normalize = normalize
        self.delete_background = delete_background

    def to_array(self, example):
        filename = os.path.expanduser(example['filename'])
        box = example.get('box') if self.bounding_box else None
        img = imutil.decode_jpg(filename, 
                resize_to=self.img_shape, 
                crop_to_box=box)
        if self.delete_background:
            seg_filename = os.path.expanduser(example['segmentation'])
            segmentation = imutil.decode_jpg(seg_filename,
                    resize_to=self.img_shape,
                    crop_to_box=box)
            foreground_mask = segmentation[:,:,0] > 0
            img = img * np.expand_dims(foreground_mask, axis=-1)
        if self.random_horizontal_flip and random.getrandbits(1):
            img = np.flip(img, axis=1)
        if self.torch:
            img = img.transpose((2,0,1))
        if self.normalize:
            img *= 1.0 / 255
        return img

    def from_array(self, array):
        return array


# LabelConverter converts eg. the MNIST label "2" to the one-hot [00100000000]
class LabelConverter(Converter):
    def __init__(self, dataset, label_key="label", **kwargs):
        self.label_key = label_key
        unique_labels = set()
        for example in dataset.examples:
            label = example.get(label_key)
            unique_labels.add(label)
        self.labels = sorted(list(unique_labels))
        self.num_classes = len(self.labels)
        self.idx = {self.labels[i]: i for i in range(self.num_classes)}

    def to_array(self, example):
        return self.idx[example[self.label_key]]

    def from_array(self, array):
        return str(np.argmax(array))

    def to_categorical(self, idx):
        res = np.zeros(self.num_classes)
        res[idx] = 1
        return res
