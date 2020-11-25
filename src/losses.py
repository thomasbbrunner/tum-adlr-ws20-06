import torch
from torch import nn
import torch.nn.functional as F

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
    recon_loss = F.mse_loss(recon_x, x)
    # recon_loss = F.hinge_embedding_loss(recon_x, x)

    # KL-divergence between the prior distribution over latent vectors
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + variational_beta * kldivergence

