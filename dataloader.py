import torch
from dataset_file import DatasetFile
from converter import ImageConverter, LabelConverter
from torchvision import transforms


class CustomDataloader(object):
    def __init__(self, dataset='mnist.dataset', batch_size=16, fold='train', **kwargs):
        self.dsf = DatasetFile(dataset)
        self.img_conv = ImageConverter(self.dsf, **kwargs)
        self.lab_conv = LabelConverter(self.dsf, **kwargs)
        self.batch_size = batch_size
        self.fold = fold
        self.num_classes = self.lab_conv.num_classes

    def get_batch(self):
        batch = self.dsf.get_batch(fold=self.fold, batch_size=self.batch_size)
        images = torch.FloatTensor(self.img_conv(batch)).cuda()
        labels = torch.LongTensor(self.lab_conv(batch)).cuda()
        return images, labels

    def __iter__(self):
        # TODO: Yield each item once, in shuffled order
        for _ in range(len(self)):
            yield self.get_batch()

    def __len__(self):
        # TODO: Count only the items in self.fold
        return self.dsf.count() // self.batch_size
