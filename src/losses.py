import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from utils import *

'''
Maximum Mean Discrepancy (MMD)
source: https://github.com/masa-su/pixyz/blob/master/pixyz/losses/mmd.py
'''
# def MMD_tmp(x, y, device):
# produces instability
def MMD2(x, y, device):

    # if torch.any(torch.isnan(x)):
    #     raise Exception('NaN in x')
    #
    # if torch.any(torch.isnan(y)):
    #     raise Exception('NaN in y')

    def inverse_multiquadratic_kernel(x, y):

        if torch.any(torch.isnan(x)):
            raise Exception('NaN in x')
        if torch.any(torch.isinf(x)):
            raise Exception('Inf in x')

        if torch.any(torch.isnan(y)):
            raise Exception('NaN in y')
        if torch.any(torch.isinf(y)):
            raise Exception('Inf in y')

        h = 1.2

        cdist = torch.cdist(x, y, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')

        if torch.any(torch.isnan(cdist)):
            raise Exception('NaN in cdist')
        if torch.any(torch.isinf(cdist)):
            raise Exception('Inf in cdist')

        return h ** 2 / (h ** 2 + cdist)

    xx = inverse_multiquadratic_kernel(x, x)
    xy = inverse_multiquadratic_kernel(x, y)
    yy = inverse_multiquadratic_kernel(y, y)

    # if torch.any(torch.isnan(xx)):
    #     raise Exception('NaN in xx')
    #
    # if torch.any(torch.isnan(xy)):
    #     raise Exception('NaN in xy')
    #
    # if torch.any(torch.isnan(xy)):
    #     raise Exception('NaN in xy')

    mean = torch.mean(xx + yy - 2.0 * xy)

    # if torch.isnan(mean):
    #     raise Exception('NaN in MMD')

    return mean

'''
Maximum Mean Discrepancy (MMD) Multiscale
source: https://github.com/VLL-HD/analyzing_inverse_problems/blob/master/toy_8-modes/toy_8-modes.ipynb
'''
def MMD(x, y, device):

    xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*zz

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    for a in [0.05, 0.2, 0.9]:
        XX += a**2 * (a**2 + dxx)**-1
        YY += a**2 * (a**2 + dyy)**-1
        XY += a**2 * (a**2 + dxy)**-1

    mean = torch.mean(XX + YY - 2.*XY)

    return mean

'''
Mean Squared Error (MSE)
'''
def MSE(_y, y, reduction):
    return F.mse_loss(_y, y, reduction=reduction)

'''
Kullback-Leibler Divergence
'''
def KL_divergence(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return - 0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())

# # The CVAE is trained by maximizing ELBO on the marginal log-likelihood
# # Optimization of single sample Monte-Carlo estimate
# def VAE_loss(_y, y):
#     logpx_z = MSE(_y, y, reduction='sum')
#     logpz =
#

'''
Binary Cross Entropy
'''
def Binary_CE(recon_x, x):
    return F.binary_cross_entropy(recon_x, x, size_average=False)

'''
This loss is created for normalized inputs with sigmoid as the activation function in the last layer of the decoder
'''
def VAE_loss_MNIST(recon_x, x, mu, logvar, variational_beta):

    # recon_x is the probability of a multivariate Bernoulli distribution p.
    # -log(p(x)) is then the pixel-wise binary cross-entropy.
    # recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    recon_loss = F.binary_cross_entropy(recon_x, x, size_average=False)

    # KL-divergence between the prior distribution over latent vectors
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence

'''
This loss is created for unnormalized inputs without activation function in the last layer of the decoder
'''
def VAE_loss_ROBOT_SIM(recon_x, x, mu, logvar, variational_beta):

    # try out what is better
    # recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL-divergence between the prior distribution over latent vectors
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence

# Input x: gt input vector with shape: num_samples x 2 * num_joints
# Input _x: predicted input vector with shape: num_samples x 2 * num_joints
def custom_loss(_x, x, reduction):

    if torch.any(torch.isnan(_x)):
        raise Exception('NaN in _x detected!')
    if torch.any(torch.isinf(_x)):
        raise Exception('Inf in _x detected!')

    num_joints = int(x.size()[1] / 2)
    samples = x.size()[0]

    # vectorize input vectors
    _x_vectorized = vectorize(_x)
    if torch.any(torch.isnan(_x_vectorized)):
        raise Exception('NaN in _x_vectorized detected!')
    if torch.any(torch.isinf(_x_vectorized)):
        raise Exception('Inf in _x_vectorized detected!')
    x_vectorized = vectorize(x)
    if torch.any(torch.isnan(x_vectorized)):
        raise Exception('NaN in x_vectorized detected!')
    if torch.any(torch.isinf(x_vectorized)):
        raise Exception('Inf in x_vectorized detected!')

    # normalize preds direction vectors
    # eps important in order to avoid dividing by 0
    _x_normalized = nn.functional.normalize(input=_x_vectorized, p=2, dim=2, eps=1e-5)

    if torch.any(torch.isnan(_x_normalized)):
        raise Exception('NaN in _x_normalized detected!')
    if torch.any(torch.isinf(_x_normalized)):
        raise Exception('Inf in _x_normalized detected!')

    ones = torch.ones(size=(samples,))
    loss_i = 0.0
    for i in range(num_joints):
        dot_product = _x_normalized[:, i, 0] * x_vectorized[:, i, 0] + _x_normalized[:, i, 1] * x_vectorized[:, i, 1]
        loss_i += MSE(dot_product, ones, reduction=reduction)
        if torch.isnan(loss_i):
            raise Exception('NaN in loss_i detected!')

    # divide by num_joints
    loss_i /= num_joints

    return loss_i

def custom_loss2(_x, x):
    loss = MSE(_x, x, reduction='mean')
    return loss

# Testing example
if __name__ == '__main__':

    use_gpu = False
    num_samples = 4
    latent_dim = 2

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")

    x = torch.randn(num_samples, latent_dim, device=device)
    _x = torch.randn(num_samples, latent_dim, device=device)


