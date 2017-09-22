import torch
from dataset_file import DatasetFile
from converter import ImageConverter, LabelConverter
from torchvision import transforms


class CustomDataloader(object):
    def __init__(self, dataset_file, batch_size, fold=None, **kwargs):
        self.dsf = DatasetFile(dataset_file)
        self.img_conv = ImageConverter(self.dsf, **kwargs)
        self.lab_conv = LabelConverter(self.dsf)
        self.batch_size = batch_size
        self.fold = fold
        self.num_classes = self.lab_conv.num_classes

    def __iter__(self):
        for _ in range(len(self)):
            batch = self.dsf.get_batch(fold=self.fold, batch_size=self.batch_size)
            images = torch.FloatTensor(self.img_conv(batch)).cuda()
            labels = torch.LongTensor(self.lab_conv(batch)).cuda()
            yield images, labels

    def __len__(self):
        return self.dsf.count() / self.batch_size
