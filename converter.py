"""
A Converter converts between:
    examples (each one a dict with keys like "filename" and "label")
    arrays (numpy arrays input to or output from a network)

Dataset augmentation can be accomplished with a Converter that returns a
different array each time to_array is called with the same example
"""
import os
import numpy as np
import imutil


# Converters can be used like a function
class Converter(object):
    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            return [self.from_array(e) for e in inputs]
        elif isinstance(inputs, list):
            return np.array([self.to_array(e) for e in inputs])
        else:
            return self.to_array(inputs)


# Outputs images as eg. 3x32x32 FloatTensor Variables
class ImageConverter(Converter):
    def __init__(self, dataset, width=32, height=32, bounding_box=False):
        self.img_shape = (width, height)
        self.bounding_box = bounding_box
        self.data_dir = dataset.data_dir

    def to_array(self, example):
        filename = os.path.join(self.data_dir, str(example['filename']))
        box = example.get('box') if self.bounding_box else None
        img = imutil.decode_jpg(filename, 
                resize_to=self.img_shape, 
                crop_to_box=box)
        # Normalize pixels to the mean
        EPSILON = .001
        img /= (EPSILON + img.mean(axis=(0,1)))
        img -= img.mean(axis=(0,1))
        img /= 2
        return img.transpose((2,0,1))

    def from_array(self, array):
        return array


# LabelConverter converts eg. the MNIST label "2" to the one-hot [00100000000]
class LabelConverter(Converter):
    def __init__(self, dataset, label_key="label"):
        self.label_key = label_key
        unique_labels = set()
        for example in dataset.examples:
            label = example.get(label_key)
            unique_labels.add(label)
        self.labels = sorted(list(unique_labels))
        self.idx = {self.labels[i]: i for i in range(len(self.labels))}

    def to_array(self, example):
        idx = self.idx[example[self.label_key]]
        return idx

    def from_array(self, array):
        return str(np.argmax(array))
