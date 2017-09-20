import torch
from dataset_file import DatasetFile
from converter import ImageConverter
from torchvision import transforms


class CustomDataloader(object):
    def __init__(self, dataset_file, batch_size, fold=None, **kwargs):
        self.dsf = DatasetFile(dataset_file)
        self.conv = ImageConverter(self.dsf, **kwargs)
        self.batch_size = batch_size
        self.fold = fold

    def __iter__(self):
        for _ in range(len(self)):
            batch = self.dsf.get_batch(fold=self.fold, batch_size=self.batch_size)
            yield torch.FloatTensor(self.conv(batch)).cuda()

    def __len__(self):
        return self.dsf.count() / self.batch_size
