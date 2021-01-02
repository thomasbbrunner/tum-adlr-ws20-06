import torch
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from robotsim_dataset import RobotSimDataset
import robotsim

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

    # x = normalize(x)

    num_joints = int(x.size()[1] / 2)
    _x = torch.zeros(size=(x.size()[0], num_joints))

    for i in range(num_joints):
        _x[:, i] = torch.atan2(input=x[:, i], other=x[:, num_joints+i])

    return _x

def normalize(x):

    num_joints = int(x.size()[1] / 2)

    # to avoid dividing by 0
    eps = 0.00001

    for i in range(num_joints):
        length = torch.sqrt(torch.square(x[:, i]) + torch.square(x[:, num_joints + i]))
        x[:, i] = x[:, i] / (length + eps)
        x[:, num_joints + i] = x[:, num_joints + i] / (length + eps)

    return x

def plot_contour_lines(config, points, gt, percentile=0.97):

    # q_quantile = np.quantile(points, q=percentile, axis=0)
    distance = np.linalg.norm(points - gt, axis=1)

    # sorts distance array such that the first k elements are the smallest
    samples = points.shape[0]
    k = int(percentile * samples)
    # print('k: ', k)
    idx = np.argpartition(distance, k)
    idx_k = idx[:k]

    # selects
    selected_points = torch.index_select(input=torch.Tensor(points), dim=0, index=torch.LongTensor(idx_k)).detach()
    # print('SHAPE OF selected points: ', selected_points.shape)
    selected_points = np.insert(selected_points, [1], gt, axis=0)
    # print('SHAPE OF points: ', points.shape)

    hull = ConvexHull(selected_points)
    area = np.around(hull.area, decimals=2)

    fig = plt.figure()
    plt.title('Area of convex hull: ' + str(area))
    plt.axis([6.0, 9.5, -4.0, 4.0])
    plt.scatter(points[:, 0], points[:, 1], c='g')
    plt.scatter(gt[0], gt[1], c='r')
    # plt.scatter(selected_points[:, 0], selected_points[:, 1], c='b')
    for simplex in hull.simplices:
        plt.plot(selected_points[simplex, 0], selected_points[simplex, 1], 'k-')
    plt.savefig('figures/q_quantile_of_points_' + config['name'] + '_' + config['dof'] + '.png')

def RMSE(pred, gt):
    return torch.sqrt(torch.mean(torch.sum(torch.square(pred - gt), dim=1)))

def rejection_sampling(robot, tcp, dof, samples):

    eps = 0.05
    hit = 0
    hit_samples = []
    np_tcp = np.array(tcp)

    while(hit<samples):

        sampled_joints = np.random.normal(loc=0.0, scale=0.5, size=(1, dof))[0]
        sampled_tcp = robot.forward(joint_states=sampled_joints)
        sampled_tcp = [sampled_tcp[0], sampled_tcp[1]]
        sampled_tcp = np.array(sampled_tcp)
        norm = np.linalg.norm(sampled_tcp-np_tcp)

        if(norm < eps):
            hit_samples.append(sampled_joints)
            hit = hit + 1
            print('hit: ', hit)

    return hit_samples


# Testing example
if __name__ == '__main__':

    # use_gpu = False
    # num_samples = 100
    # percentile = 0.97
    #
    # device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    #
    # # x = torch.randn(num_samples, 1, device=device)
    # # y = torch.randn(num_samples, 1, device=device)
    # #
    # # points = torch.cat((x, y), dim=1)
    # # numpy_points = points.numpy()
    #
    # points = np.random.rand(num_samples, 2)
    #
    # # plot_contour_lines(points, percentile=percentile)
    #
    # pred = torch.rand(size=(num_samples, 3), device=device)
    # gt = torch.rand(size=(num_samples, 3), device=device)
    # rmse = RMSE(pred, gt)
    # print(rmse)

    robot = robotsim.Robot2D3DoF([3, 2, 3])
    gt_tcp = [6.0, 5.0]
    joint_states = rejection_sampling(robot=robot, tcp=gt_tcp, dof=3)

    joint_states = np.array(joint_states)
    robot.plot(joint_states=joint_states, separate_plots=False)
    plt.show()
