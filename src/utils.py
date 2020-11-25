import torch
import yaml
import os

# Function to load yaml configuration file
def load_config(config_name, config_path):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)
    return config

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

    # see values as complex numbers from which we want to compute its argument
    # arg(x + i*y) = arg(cos(phi) + i*sin(phi)) = arctan(y/x)
    # special cases

    num_joints = int(x.size()[1] / 2)
    _x = torch.zeros(size=(x.size()[0], num_joints))

    for i in range(num_joints):
        _x[:, i] = torch.atan2(input=x[:, i], other=x[:, num_joints+i])

    return _x