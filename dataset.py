import torch
from torch.utils import data
import numpy as np


class DataSet(torch.utils.data.Dataset):
    def __init__(self, root):
        super(DataSet, self).__init__()
        self.imgs = []
        fh = open(root + r'dataset.txt', 'r')
        lines = fh.readlines()
        for line in lines:
            label, matrix = line.strip().split()
            self.imgs.append((label, matrix))
        self.root = root

    def __getitem__(self, index):
        label, picture_name = self.imgs[index]
        path = self.root + label + r'\\' + picture_name
        img = np.load(path).reshape((1, 16, 16, 16))
        img = torch.tensor(img)
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs)

