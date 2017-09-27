import math
import torch
from dataset_file import DatasetFile
from converter import ImageConverter, LabelConverter
from torchvision import transforms


class CustomDataloader(object):
    def __init__(self, dataset='mnist.dataset', batch_size=16, fold='train', shuffle=True, **kwargs):
        self.dsf = DatasetFile(dataset)
        self.img_conv = ImageConverter(self.dsf, **kwargs)
        self.lab_conv = LabelConverter(self.dsf, **kwargs)
        self.batch_size = batch_size
        self.fold = fold
        self.shuffle = shuffle
        self.num_classes = self.lab_conv.num_classes

    def get_batch(self):
        batch = self.dsf.get_batch(fold=self.fold, batch_size=self.batch_size)
        images = torch.FloatTensor(self.img_conv(batch)).cuda()
        labels = torch.LongTensor(self.lab_conv(batch)).cuda()
        return images, labels

    def __iter__(self):
        for batch in self.dsf.get_all_batches(fold=self.fold, batch_size=self.batch_size, shuffle=self.shuffle):
            images = torch.FloatTensor(self.img_conv(batch)).cuda()
            labels = torch.LongTensor(self.lab_conv(batch)).cuda()
            yield images, labels

    def __len__(self):
        return math.floor(self.dsf.count(self.fold) / self.batch_size)
