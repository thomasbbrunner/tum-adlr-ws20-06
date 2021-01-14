import torch
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d

from robotsim_dataset import RobotSimDataset
import robotsim
import robotsim_plot

'''
Various methods to make life easier
'''

# Function to load yaml configuration file
def load_config(config_name, config_path):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)
    return config

# def onehot(idx, num_classes):
#
#     assert idx.shape[1] == 1
#     assert torch.max(idx).item() < num_classes
#
#     onehot = torch.zeros(idx.size(0), num_classes)
#     onehot.scatter_(1, idx.data, 1)
#
#     return onehot

# Takes joints angles as inputs and produces direction vectors from them
def preprocess(x):

    x_sin = torch.sin(x)
    x_cos = torch.cos(x)
    x = torch.cat((x_sin, x_cos), dim=1)
    return x

# def postprocess_old(x):
#
#     # see values as complex numbers from which we want to compute its argument
#     # arg(x + i*y) = arg(cos(phi) + i*sin(phi)) = arctan(y/x)
#     # special cases
#
#     # x = normalize(x)
#
#     num_joints = int(x.size()[1] / 2)
#     _x = torch.zeros(size=(x.size()[0], num_joints))
#
#     for i in range(num_joints):
#         _x[:, i] = torch.atan2(input=x[:, i], other=x[:, num_joints+i])
#
#     return _x

# takes direction vectors as input and computes corresponding joint angles
def postprocess(x):

    num_joints = int(x.size()[1] / 2)
    _x = torch.zeros(size=(x.size()[0], num_joints))

    for i in range(num_joints):
        _x[:, i] = torch.atan2(input=x[:, i], other=x[:, num_joints+i])

    return _x

# Normalizes vectorized (--> see method vectorize()) direction vectors
def normalize(x):

    num_joints = int(x.size()[1] / 2)
    x_normalized = x.clone()

    # to avoid dividing by 0
    eps = 0.00001

    for i in range(num_joints):
        length = torch.sqrt(torch.square(x[:, i]) + torch.square(x[:, num_joints + i]))
        x_normalized[:, i] = x[:, i] / (length + eps)
        x_normalized[:, num_joints + i] = x[:, num_joints + i] / (length + eps)

    return x_normalized

# Takes the output x of the network which has 2*num_joints variables and returns num_joints direction vectors
# Input shape: num_samples x 2 * num_joints
# Output shape: num_samples x num_joints x 2
def vectorize(x):

    num_joints = int(x.size()[1] / 2)
    samples = x.size()[0]

    x_vectorized = torch.zeros(size=(samples, num_joints, 2))
    for i in range(num_joints):
        x_vectorized[:, i, 0] = x[:, i]
        x_vectorized[:, i, 1] = x[:, num_joints + i]

    return x_vectorized

def plot_contour_lines(points, gt, PATH, percentile=0.97):

    # q_quantile = np.quantile(points, q=percentile, axis=0)
    distance = np.linalg.norm(points - gt, axis=1)

    # sorts distance array such that the first k elements are the smallest
    samples = points.shape[0]
    k = int(percentile * samples)
    idx = np.argpartition(distance, k)
    idx_k = idx[:k]

    # selects
    selected_points = torch.index_select(input=torch.Tensor(points), dim=0, index=torch.LongTensor(idx_k)).detach()
    selected_points = np.insert(selected_points, [1], gt, axis=0)

    hull = ConvexHull(selected_points)
    area = np.around(hull.area, decimals=2)

    fig = plt.figure()
    plt.title('Area of convex hull: ' + str(area))
    # define axis limits dynamically
    plt.axis([gt[0]-3.0, gt[0]+3.0, gt[1]-3.0, gt[1]+3.0])
    plt.scatter(points[:, 0], points[:, 1], c='g')
    plt.scatter(gt[0], gt[1], c='r')

    for simplex in hull.simplices:
        plt.plot(selected_points[simplex, 0], selected_points[simplex, 1], 'k-')
    plt.savefig(PATH)

# def RMSE(pred, gt):
#     return torch.sqrt(torch.mean(torch.sum(torch.square(pred - gt), dim=1)))

# method to produce a ground truth posterior distribution of the joint angles depending on the tcp
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

    return hit_samples

# from robotsim_dataset
def plot_configurations(robot, joints, transparency=None, path=None, show=False):

    fig, ax = robotsim_plot.heatmap(joints, robot, transparency=transparency, path=None, show=False)
    plt.axis([0.0, 2.7, -1.0, 1.0])

    # get a sample to plot
    joint_coords = robot._get_joint_coords(joints[22])

    for arm in joint_coords:
        ax.plot(
            arm[:, 0].flatten(),
            arm[:, 1].flatten(),
            c='k', linewidth=2, zorder=10)

        ax.scatter(
            arm[:, 0].flatten(),
            arm[:, 1].flatten(),
            c='r', s=6, zorder=11)

    if path:
        fig.savefig(path, dpi=300)

    if show:
        fig.show()


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
    robotsim_plot.plot(joint_states, robot)
    plt.show()
