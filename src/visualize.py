import torch

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from models.CVAE import *
from utils import onehot

import matplotlib.pyplot as plt

if __name__ == '__main__':

    ####################################################################################################################
    # TO MODIFY
    ####################################################################################################################

    # 2-d latent space, parameter count in same order of magnitude
    # as in the original VAE paper (VAE paper has about 3x as many)
    X_dim = 28 * 28
    hidden_dim = 100
    latent_dim = 2
    batch_size = 128
    num_classes = 10
    # capacity = 64
    learning_rate = 1e-3
    variational_beta = 1
    use_gpu = False
    PATH = 'weights/MNIST_CVAE'

    ####################################################################################################################
    # LOAD DATASET
    ####################################################################################################################

    img_transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = MNIST(root='./data/MNIST', download=True, train=False, transform=img_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    ####################################################################################################################
    # BUILD MODEL
    ####################################################################################################################

    cvae = CVAE(X_dim, hidden_dim, latent_dim, num_classes)

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    cvae = cvae.to(device)

    cvae.load_weights(PATH)

    # set to evaluation mode
    cvae.eval()

    ####################################################################################################################
    # VISUALISATION
    ####################################################################################################################

    # create a random latent vector
    z = torch.randn(1, latent_dim).to(device)

    # pick a class, for which we want to generate the data
    # pick randomly 1 class, for which we want to generate the data
    y = torch.randint(0, num_classes, (1, 1)).to(dtype=torch.long)

    print(f'Generating a {y.item()}')

    y = onehot(y, num_classes).to(device, dtype=z.dtype)

    z = torch.cat((z, y), dim=1)

    reconstructed_img = cvae.decoder(z)
    img = reconstructed_img.view(28, 28).data

    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()