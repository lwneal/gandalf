import math
import torch
from dataset_file import DatasetFile
from converter import ImageConverter, LabelConverter, AttributeConverter
from torchvision import transforms


class CustomDataloader(object):
    def __init__(self, dataset='mnist.dataset', batch_size=16, fold='train', shuffle=True, last_batch=False, example_count=None, **kwargs):
        self.dsf = DatasetFile(dataset, example_count=example_count)
        self.img_conv = ImageConverter(self.dsf, **kwargs)
        self.lab_conv = LabelConverter(self.dsf, **kwargs)
        self.attr_conv = AttributeConverter(self.dsf, **kwargs)
        self.batch_size = batch_size
        self.fold = fold
        self.last_batch = last_batch
        self.shuffle = shuffle
        self.num_classes = self.lab_conv.num_classes
        self.num_attributes = self.attr_conv.num_attributes

    def get_batch(self):
        batch = self.dsf.get_batch(fold=self.fold, batch_size=self.batch_size)
        images, labels, attributes = self.convert(batch)
        return images, labels, attributes

    def __iter__(self):
        batcher = self.dsf.get_all_batches(
                fold=self.fold,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                last_batch=self.last_batch)
        for batch in batcher:
            images, labels, attributes = self.convert(batch)
            yield images, labels, attributes

    def convert(self, batch):
        images = torch.FloatTensor(self.img_conv(batch)).cuda()
        labels = torch.LongTensor(self.lab_conv(batch)).cuda()
        if self.num_attributes > 0:
            attributes = torch.FloatTensor(self.attr_conv(batch)).cuda()
        else:
            attributes = None
        return images, labels, attributes

    def __len__(self):
        return math.floor(self.dsf.count(self.fold) / self.batch_size)
