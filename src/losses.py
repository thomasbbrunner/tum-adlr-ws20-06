import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from utils import *

def MSE(_y, y, reduction):
    """ Calculates the Mean squared Error (MSE) of _y and y

    Args:
        _y: predicted variable
        y: ground truth variable
        reduction: 'sum' or 'mean'

    Returns: Mean Squared Error

    """

    return F.mse_loss(_y, y, reduction=reduction)

def loss_without_singularities(_x, x, config):
    """Calculates the difference between the joint angles considering the definition boundaries of (-pi, pi)
    Example: ensures that theta1 = 178° and theta2 = -178° only have a difference of 4°

    Args:
        _x: predicted joint angles
        x: ground truth joint angles
        config: yaml configuration file

    Returns: MSE of converted joint angles

    """

    x_short = x[:, :config['input_dim']]
    _x_short = _x[:, :config['input_dim']]

    angle2minuspi_pluspi = torch.atan2(torch.sin(_x_short - x_short), torch.cos(_x_short - x_short))

    #  TODO: mean or sum as reduction ?
    d_x = torch.sum(angle2minuspi_pluspi**2)
    # consider possible padding in INN model
    d_pad = F.mse_loss(_x[:, config['input_dim']:], x[:, config['input_dim']:], reduction='sum')
    return d_x + d_pad

def custom_loss(_x, x, config):

    """Calculates the difference of the joint angles in vector-based representation

    Goal (_v: predicted direction vector, v: gt direction vector):
    _v and v are the same if loss(_v dot product v, 1) == 0

    Args:
        _x: predicted tensor of shape num_samples * [sin(theta1), sin(theta2), ..., cos(theta1), cos(theta2), ...]
        x: predicted tensor of shape num_samples * [sin(theta1), sin(theta2), ..., cos(theta1), cos(theta2), ...]
        config: yaml configuration file

    Returns: accumulated loss of the joint angles in vector-based representation divided by the number of joint angles

    """

    num_joints = config['dof']
    samples = x.size()[0]
    x_short = x[:, :config['input_dim']]
    _x_short = _x[:, :config['input_dim']]

    # vectorize input vectors
    _x_vectorized = vectorize(_x_short)
    x_vectorized = vectorize(x_short)

    # normalize predicted direction vectors
    _x_normalized = nn.functional.normalize(input=_x_vectorized, p=2, dim=2, eps=1e-5) # eps: avoid dividing by 0

    ones = torch.ones(size=(samples,))
    loss_i = 0.0
    for i in range(num_joints):
        dot_product = _x_normalized[:, i, 0] * x_vectorized[:, i, 0] + _x_normalized[:, i, 1] * x_vectorized[:, i, 1]
        loss_i += MSE(dot_product, ones, reduction='sum')

    # divide by num_joints
    loss_i /= num_joints
    # consider possible padding in INN model
    loss_pad = F.mse_loss(_x[:, config['input_dim']:], x[:, config['input_dim']:], reduction='sum')
    return loss_i + loss_pad

def MSEloss4joints(_x, x, config):
    """ Calculates the discrepancy between the predicted joint angles and the gt joint angles
    loss computation dependent on the shape of the input:
    Either vector-based representation of joint angles or direct joint angles
    In both cases, possible singularities are considered

    Args:
        _x: predicted joints
        x: ground truth joints
        config: configuration file

    Returns: loss between _x and x

    """

    if config['dof'] == config['input_dim']:
        loss = loss_without_singularities(_x, x, config=config)
        # loss = MSE(_x, x, reduction='sum')
    elif config['dof'] * 2 == config['input_dim']:
        loss = custom_loss(_x, x, config=config)
    else:
        raise Exception("Input dimension of joints invalid!")
    return loss

def MSEloss4tcp(_y, y):
    """Calculates the MSE for the tcp

    Args:
        _y: predicted (x, y) coordinates of tcp
        y: ground truth (x, y) coordinates of tcp

    Returns: Mean Squared Error between _y and y

    """

    return MSE(_y, y, reduction='sum')

def MMD(x, y):
    """Calculates the Maximum Mean Discrepancy (MMD) between two propability distributions x and y
    source: https://github.com/masa-su/pixyz/blob/master/pixyz/losses/mmd.py

    Args:
        x: probability distribution 1
        y: probability distribution 2

    Returns: value of the discrepancy between x and y

    """

    def inverse_multiquadratic_kernel(x, y):
        """Kernel function for computing the Maximum Mean Discrpancy

        Args:
            x: probability distribution 1
            y: probability distribution 2

        Returns: value of inverse multiquadratic function

        """

        h = 1.2
        cdist = torch.cdist(x, y, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
        return h ** 2 / (h ** 2 + cdist)

    xx = inverse_multiquadratic_kernel(x, x)
    xy = inverse_multiquadratic_kernel(x, y)
    yy = inverse_multiquadratic_kernel(y, y)
    return torch.mean(xx + yy - 2.0 * xy)

def KL_divergence(mu, logvar):
    """Calculates the Kullback-Leibler Divergence between probability distribution with mean mu and log variance logvar
    and N(0, 1)

    see Appendix B from VAE paper:
    Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114

    Args:
        mu: predicted mean of the probability distribution
        logvar: predicted log variance of the probability distribution

    Returns: KL-divergence

    """

    return - 0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())

# TODO: Implement Tests