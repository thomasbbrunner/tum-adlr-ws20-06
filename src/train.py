import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from models.CVAE import *
from losses import VAE_loss
from utils import onehot


if __name__ == '__main__':

    ####################################################################################################################
    # TO MODIFY
    ####################################################################################################################

    # 2-d latent space, parameter count in same order of magnitude
    # as in the original VAE paper (VAE paper has about 3x as many)
    X_dim = 28*28
    hidden_dim = 100
    latent_dim = 2
    num_classes = 10
    num_epochs = 30
    batch_size = 128
    # capacity = 64
    learning_rate = 1e-3
    variational_beta = 1
    use_gpu = False
    PATH = 'weights/MNIST_CVAE'

    ####################################################################################################################
    # LOAD DATASET
    ####################################################################################################################

    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = MNIST(root='./data/MNIST', download=True, train=True, transform=img_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ####################################################################################################################
    # BUILD MODEL
    ####################################################################################################################

    cvae = CVAE(X_dim, hidden_dim, latent_dim, num_classes)

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    vae = cvae.to(device)

    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print('Number of parameters: %d' % num_params)

    optimizer = torch.optim.Adam(params=vae.parameters(), lr=learning_rate, weight_decay=1e-5)

    ####################################################################################################################
    # TRAINING
    ####################################################################################################################

    # set to training mode
    vae.train()

    train_loss_avg = []

    print('Training ...')
    for epoch in range(num_epochs):

        train_loss_avg.append(0)
        num_batches = 0

        for image_batch, label_batch in train_dataloader:
            image_batch = image_batch.to(device)

            # reshape the data into [batch_size, 784]
            image_batch = image_batch.view(image_batch.size(0), -1)

            # convert y into one-hot encoding
            label_batch = onehot(label_batch.view(-1, 1), num_classes)

            # vae reconstruction
            image_batch_recon, latent_mu, latent_logvar = cvae(image_batch, label_batch)

            # reconstruction error
            loss = VAE_loss(image_batch_recon, image_batch, latent_mu, latent_logvar, variational_beta)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # one step of the optimizer (using the gradients from backpropagation)
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1

        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch + 1, num_epochs, train_loss_avg[-1]))

    vae.save_weights(PATH)