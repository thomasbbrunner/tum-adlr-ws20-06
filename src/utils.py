import torch

def onehot(idx, num_classes):

    assert idx.shape[1] == 1
    assert torch.max(idx).item() < num_classes

    onehot = torch.zeros(idx.size(0), num_classes)
    onehot.scatter_(1, idx.data, 1)

    return onehot

def preprocess(x):

    x_sin = torch.sin(x)
    x_cos = torch.cos(x)
    x = torch.cat((x_sin, x_cos), dim=1)
    return x

def postprocess(x):

    # print('SHAPE: ', x.size())
    x_arcsin = torch.arcsin(x[:, :3])
    x_arccos = torch.arccos(x[:, 3:])
    x = torch.cat((x_arcsin, x_arccos), dim=1)
    return x