import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn

# MMD is a distance-based measure between two distributions p and q based on the mean embeddings mu_p and mu_q
# in a reproducing kernel Hilbert space F:
# MMD(F, p, q) = || mu_p - mu_q || **2
def MMD_loss(x, y, device):

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

    return torch.mean(XX + YY - 2.*XY)

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

def INN_loss_ROBOT_SIM(output, y, output_inv, x, config, device):

    # L_y
    # For forward iteration, the deviation between simulation outcomes and network predictions are penalized
    L_y = MSELoss(output[:, :config['output_dim']], y[:, :config['output_dim']])
    # print('L_y: ', L_y)

    # L_z
    # Loss for latent variable computed by Maximum Mean Discrepancy (MMD)
    # Penalizes mismatch between joint distribution of network outputs and the product of marginal distributions of
    # simulation outcomes and latents
    L_z = MMD_loss(output[:, config['output_dim']:], y[:, config['output_dim']:], device)
    # print('L_z: ', L_z)

    # L_x
    L_x = MMD_loss(output_inv, x, device)
    # print('L_x: ', L_x)

    return config['weight_Ly'] * L_y + config['weight_Lz'] * L_z + config['weight_Lx'] * L_x

def MSELoss(_y, y):
    return F.mse_loss(_y, y, reduction='sum')