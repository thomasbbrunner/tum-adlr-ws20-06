import torch
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d

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

def plot_contour_lines(points, percentile=0.97):

    q_quantile = np.quantile(points, q=percentile, axis=0)
    distance = np.linalg.norm(points - q_quantile, axis=1)

    # sorts distance array such that the first k elements are the smallest
    samples = points.shape[0]
    k = int(percentile * samples)
    # print('k: ', k)
    idx = np.argpartition(distance, k)
    idx_k = idx[:k]

    # selects
    selected_points = torch.index_select(input=torch.Tensor(points), dim=0, index=torch.LongTensor(idx_k)).numpy()
    hull = ConvexHull(selected_points)

    fig = plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c='g')
    # plt.scatter(q_quantile[0], q_quantile[1], c='r')
    # plt.scatter(selected_points[:, 0], selected_points[:, 1], c='b')
    for simplex in hull.simplices:
        plt.plot(selected_points[simplex, 0], selected_points[simplex, 1], 'k-')
    plt.show()
    # plt.savefif(PATH + '')

# Testing example
if __name__ == '__main__':

    use_gpu = False
    num_samples = 100
    percentile = 0.97

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    # x = torch.randn(num_samples, 1, device=device)
    # y = torch.randn(num_samples, 1, device=device)
    #
    # points = torch.cat((x, y), dim=1)
    # numpy_points = points.numpy()

    points = np.random.rand(num_samples, 2)

    plot_contour_lines(points, percentile=percentile)

