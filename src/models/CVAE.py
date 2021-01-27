import torch
from torch import nn
import torch.nn.functional as F
from utils import *

class Encoder(nn.Module):
    """Encoder network"""

    def __init__(self, X_dim, hidden_dim, latent_dim, num_cond, num_layers):

        super(Encoder, self).__init__()
        # create list of hidden layers
        self.fcs = nn.ModuleList([nn.Linear(in_features=X_dim + num_cond, out_features=hidden_dim)])
        self.fcs.extend([nn.Linear(in_features=hidden_dim, out_features=hidden_dim) for i in range(1, num_layers)])
        # mean of latent space
        self.fc_mu = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        # deviation of latent space
        self.fc_logvar = nn.Linear(in_features=hidden_dim, out_features=latent_dim)

    def forward(self, x):
        for layer in self.fcs:
            x = F.relu(layer(x))
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    """Decoder network"""

    def __init__(self, X_dim, hidden_dim, latent_dim, num_cond, num_layers):

        super(Decoder, self).__init__()
        # create list of hidden layers
        self.fcs = nn.ModuleList([nn.Linear(in_features=latent_dim + num_cond, out_features=hidden_dim)])
        self.fcs.extend([nn.Linear(in_features=hidden_dim, out_features=hidden_dim) for i in range(1, num_layers)])
        self.fc_output = nn.Linear(in_features=hidden_dim, out_features=X_dim)

    def forward(self, x):
        for layer in self.fcs:
            x = F.relu(layer(x))
        x = self.fc_output(x)
        return x

class CVAE(nn.Module):
    """Conditional Autoencoder with fully connected layers
    The condition (x,y) is concatenated both with the input space X and the latent space z

    cVAE paper:  Learning structured output representation using deep conditional generative models [Sohn et al 2015]
    Code inspired from source: https://github.com/graviraja/pytorch-sample-codes/blob/master/conditional_vae.py
    """

    def __init__(self, config):

        super(CVAE, self).__init__()

        self.latent_dim = config['latent_dim']
        self.X_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.num_condition = config['condition_dim']
        self.encoder = Encoder(self.X_dim, self.hidden_dim, self.latent_dim, self.num_condition, self.num_layers)
        self.decoder = Decoder(self.X_dim, self.hidden_dim, self.latent_dim, self.num_condition, self.num_layers)


    def forward(self, x, condition):
        x = torch.cat((x, condition), dim=1)
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        z = torch.cat((latent, condition), dim=1)
        x_recon = self.decoder(z)
        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        """Samples from a normal distribution with mean mu and log variance logvar

        Args:
            mu: mean
            logvar: log variance

        Returns: samples from N(mu, exp(logvar))

        """

        # the reparameterization trick
        std = logvar.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()
        return eps.mul(std).add_(mu)

    def predict(self, tcp, device):
        """Predicts joint angles dependent on the tcp + by sampling from N(0, 1)

        Args:
            tcp: (x, y) coordinates of end-effector
            device: 'cpu' or 'gpu'

        Returns: predicted joint angles

        """

        # Sample z from standard normal distribution
        z = torch.randn(tcp.size()[0], self.latent_dim, device=device)
        x = torch.cat((z, tcp), dim=1)
        with torch.no_grad():
            x = self.decoder(x)
        return x

    def save_checkpoint(self, epoch, optimizer, loss, PATH):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, PATH)

    def load_checkpoint(self, PATH, optimizer=None):
        if torch.cuda.is_available():
            checkpoint = torch.load(PATH)
        else:
            checkpoint = torch.load(PATH, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        if not optimizer == None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return optimizer, epoch, loss
        else:
            return epoch, loss

    def save_weights(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load_weights(self, PATH):
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(PATH))
        else:
            self.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))