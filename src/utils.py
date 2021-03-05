import torch
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import torch.nn.functional as F

import robotsim

def load_config(config_file):
    """Loads config file. 
    If the file can't be found, the function looks in the 
    .config/ directory.
    """

    path = pathlib.Path(config_file)
    if not path.exists():
        # look in ./config folder
        path = pathlib.Path("config", config_file)
        if not path.exists():
            raise ValueError(
                "Could not find config file: {}".format(config_file))

    with open(path) as fd:
        config = yaml.safe_load(fd)

    return config

def preprocess(x, config):
    """Preprocess the joint angles depending on the used representation (direct or vector-based)

    Args:
        x: joint angles
        config: yaml config file

    Returns: the intial joint angles or the vector-based representation of them

    """

    if config['dof'] == config['input_dim']:
        pass
    elif config['dof'] * 2 == config['input_dim']:
        x_sin = torch.sin(x)
        x_cos = torch.cos(x)
        x = torch.cat((x_sin, x_cos), dim=1)
    else:
        raise Exception("Input dimension of joints invalid!")

    return x

def postprocess(x, config):
    """Transforms the input to direct joint angles again

    Args:
        x: direct joint angles or vector-based representation of them
        config: yaml config file

    Returns: Joint angles

    """

    x_short = x[:, :config['input_dim']]

    if config['dof'] == config['input_dim']:
        _x = x_short
    elif config['dof'] * 2 == config['input_dim']:
        num_joints = config['dof']
        _x = torch.zeros(size=(x_short.size()[0], num_joints))

        for i in range(num_joints):
            _x[:, i] = torch.atan2(input=x_short[:, i], other=x_short[:, num_joints + i])
    else:
        raise Exception("Input dimension of joints invalid!")

    return _x

def vectorize(x):
    """Takes the output x of the network which has 2*num_joints variables and returns num_joints direction vectors

    Args:
        x: input of shape num_samples x 2 * num_joints

    Returns: vectorized tensor of shape num_samples x num_joints x 2

    """

    num_joints = int(x.size()[1] / 2)
    samples = x.size()[0]

    x_vectorized = torch.zeros(size=(samples, num_joints, 2))
    for i in range(num_joints):
        x_vectorized[:, i, 0] = x[:, i]
        x_vectorized[:, i, 1] = x[:, num_joints + i]

    return x_vectorized

def normalize(x):
    """Normalizes the vectorized direction vectors (--> see vectorize() method) of the corresponding joint angles

    Args:
        x: Tensor of direction vector of shape
        num_samples * num_joints * 2( --> sin(theta), cos(theta))

        Example: sin(theta1)**2 + cos(theta1)**2  =   1.0

    Returns: Tensor normalized wrt the respective joint angles

    """

    num_joints = int(x.size()[1] / 2)
    x_normalized = x.clone()

    # to avoid dividing by 0
    eps = 0.00001

    for i in range(num_joints):
        length = torch.sqrt(torch.square(x[:, i]) + torch.square(x[:, num_joints + i]))
        x_normalized[:, i] = x[:, i] / (length + eps)
        x_normalized[:, num_joints + i] = x[:, num_joints + i] / (length + eps)

    return x_normalized

def plot_contour_lines(points, gt, PATH, title="", percentile=0.97):
    """Draws a convex hull around the points which are the nearest in a percentile

    Args:
        points: list of (x, y) coordinates of points
        gt: (x, y) coordinate of ground truth point
        PATH: path to directory where figure is stored in
        percentile: between 0.0 and 1.0

    """

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
    plt.suptitle(title)
    plt.grid(True, color="#93a1a1", alpha=0.3)
    # define axis limits dynamically
    plt.axis([gt[0]-0.15, gt[0]+0.15, gt[1]-0.15, gt[1]+0.15])
    plt.scatter(points[:, 0], points[:, 1], c='g', s=0.5)
    plt.scatter(gt[0], gt[1], c='r')
    for simplex in hull.simplices:
        plt.plot(selected_points[simplex, 0], selected_points[simplex, 1], 'k-')
    plt.savefig(PATH)

def rejection_sampling(robot, tcp, dof, samples):
    """method to produce a ground truth posterior distribution of the joint angles depending on the tcp"""

    raise RuntimeError(
        "Deprecated. Use the rejection_sampling method in the robotsim package instead.")

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

def plot_configurations(robot, joints, transparency=None, path=None, show=False):
    """from robotsim_dataset"""

    raise RuntimeError(
        "Deprecated. Use the heatmap method in the robotsim package instead.")

    fig, ax = robotsim.heatmap(joints, robot, transparency=transparency, path=None, show=False)
    plt.axis([0.0, 2.7, -1.0, 1.0])

    # get a sample to plot
    joint_coords = robot.get_joint_coords(joints[22])

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

def dh_transformation(alpha, a, d, theta, squeeze=True):
    """Returns transformation matrix between two frames
    according to the Denavit-Hartenberg convention presented
    in 'Introduction to Robotics' by Craig.

    Also accepts batch processing of several joint states (d or theta).

    Transformation from frame i to frame i-1:
    alpha:  alpha_{i-1}
    a:      a_{i-1}
    d:      d_i     (variable in prismatic joints)
    theta:  theta_i (variable in revolute joints)
    """

    d = np.atleast_1d(d)
    theta = np.atleast_1d(theta)

    if d.shape[0] > 1 and theta.shape[0] > 1:
        raise RuntimeError(
            "Only one variable joint state is allowed.")

    desired_shape = np.maximum(d.shape[0], theta.shape[0])

    alpha = np.resize(alpha, desired_shape)
    a = np.resize(a, desired_shape)
    d = np.resize(d, desired_shape)
    theta = np.resize(theta, desired_shape)
    zeros = np.zeros(desired_shape)
    ones = np.ones(desired_shape)

    sin = np.sin
    cos = np.cos
    th = theta
    al = alpha

    transformation = np.array([
        [cos(th),           -sin(th),           zeros,          a],
        [sin(th)*cos(al),   cos(th) * cos(al),  -sin(al),   -sin(al)*d],
        [sin(th)*sin(al),   cos(th) * sin(al),  cos(al),    cos(al)*d],
        [zeros,             zeros,              zeros,      ones]
    ])

    # fix dimensions
    transformation = np.rollaxis(transformation, 2)

    if squeeze:
        transformation = np.squeeze(transformation)

    return transformation

def wrap(angles):
    """Wraps angles to [-pi, pi) range."""
    # wrap angles to range [-pi, pi)
    return (angles + np.pi) % (2*np.pi) - np.pi

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

    # robot = robotsim.Robot2D3DoF([3, 2, 3])
    # gt_tcp = [6.0, 5.0]
    # joint_states = rejection_sampling(robot=robot, tcp=gt_tcp, dof=3)
    #
    # joint_states = np.array(joint_states)
    # robotsim_plot.plot(joint_states, robot)
    # plt.show()

    pass
