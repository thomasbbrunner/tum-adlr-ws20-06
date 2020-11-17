import torch
from torch import nn
import torch.nn.functional as F

def onehot(idx, num_classes):

    assert idx.shape[1] == 1
    assert torch.max(idx).item() < num_classes

    onehot = torch.zeros(idx.size(0), num_classes)
    onehot.scatter_(1, idx.data, 1)

    return onehot