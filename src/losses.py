import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from utils import *

'''
Definition space of torch.atan2(y / x): x>0
'''
def MSEloss4joints(_x, x, config):

    if config['dof'] == config['input_dim']:
        loss = consider_singularities(_x, x, config=config)
        # loss = MSE(_x, x, reduction='sum')
    elif config['dof'] * 2 == config['input_dim']:
        loss = custom_loss(_x, x, config=config)
    else:
        raise Exception("Input dimension of joints invalid!")

    return loss


def MSEloss4tcp():
    pass

'''
Implementations of losses used for training the models
'''
# Maximum Mean Discrepancy (MMD)
# source: https://github.com/masa-su/pixyz/blob/master/pixyz/losses/mmd.py
def MMD(x, y, device):

    def inverse_multiquadratic_kernel(x, y):

        h = 1.2
        cdist = torch.cdist(x, y, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')

        return h ** 2 / (h ** 2 + cdist)

    xx = inverse_multiquadratic_kernel(x, x)
    xy = inverse_multiquadratic_kernel(x, y)
    yy = inverse_multiquadratic_kernel(y, y)

    return torch.mean(xx + yy - 2.0 * xy)


# Mean Squared Error (MSE)
def MSE(_y, y, reduction):
    return F.mse_loss(_y, y, reduction=reduction)


# Kullback-Leibler Divergence
def KL_divergence(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    return - 0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())


# Binary Cross Entropy
def Binary_CE(recon_x, x):
    return F.binary_cross_entropy(recon_x, x, size_average=False)

# '''
# This loss is created for unnormalized inputs without activation function in the last layer of the decoder
# '''
# def VAE_loss_ROBOT_SIM(recon_x, x, mu, logvar, variational_beta):
#
#     # try out what is better
#     # recon_loss = F.mse_loss(recon_x, x, reduction='mean')
#     recon_loss = F.mse_loss(recon_x, x, reduction='sum')
#
#     # KL-divergence between the prior distribution over latent vectors
#     kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#
#     return recon_loss + variational_beta * kldivergence

# Input x: gt input vector with shape: num_samples x 2 * num_joints
# Input _x: predicted input vector with shape: num_samples x 2 * num_joints
def custom_loss(_x, x, config):

    num_joints = config['dof']
    samples = x.size()[0]

    x_short = x[:, :config['input_dim']]
    _x_short = _x[:, :config['input_dim']]

    # vectorize input vectors
    _x_vectorized = vectorize(_x_short)
    x_vectorized = vectorize(x_short)

    # normalize preds direction vectors
    # eps important in order to avoid dividing by 0
    _x_normalized = nn.functional.normalize(input=_x_vectorized, p=2, dim=2, eps=1e-5)

    ones = torch.ones(size=(samples,))
    loss_i = 0.0
    for i in range(num_joints):
        dot_product = _x_normalized[:, i, 0] * x_vectorized[:, i, 0] + _x_normalized[:, i, 1] * x_vectorized[:, i, 1]
        loss_i += MSE(dot_product, ones, reduction='sum')

    # divide by num_joints
    loss_i /= num_joints

    loss_pad = F.mse_loss(_x[:, config['input_dim']:], x[:, config['input_dim']:], reduction='sum')

    # if config['model'] == 'INN':
    #     loss_pad = F.mse_loss(_x[:, config['input_dim']:], x[:, config['input_dim']:], reduction='sum')
    # else:
    #     loss_pad = 0.0

    return loss_i + loss_pad

if __name__ == '__main__':

    use_gpu = False
    num_samples = 4
    latent_dim = 2

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    x = torch.randn(num_samples, latent_dim, device=device)
    _x = torch.randn(num_samples, latent_dim, device=device)


