import torch
from torch import nn
import torch.nn.functional as F
from utils import *

'''
 Sources
 -------
 CVAE paper:  Learning structured output representation using deep conditional generative models [Sohn et al 2015]
 Code: https://github.com/graviraja/pytorch-sample-codes/blob/master/conditional_vae.py
 '''

class Preprocessor(nn.Module):

    def __init__(self, dof=3):
        super(Preprocessor, self).__init__()
        self.dof = dof

    def forward(self, x):
        return preprocess(x)

class Postprocessor(nn.Module):

    def __init__(self, dof=3):
        super(Postprocessor, self).__init__()
        self.dof = dof

    def forward(self, x):
        return postprocess(x)

class Encoder(nn.Module):
    def __init__(self, X_dim, hidden_dim, latent_dim, num_cond):

        '''
        Encoder network with only fully connected layers and ReLU activations

        Args:
            X_dim: number of input variables (joint angles, ect.)
            hidden_dim: number of nodes of the fully connected layers
            latent_dims: number of nodes for additional variable z)
        '''

        super(Encoder, self).__init__()

        '''
        self.fc_layers = []
        self.fc_layers.append(nn.Linear(in_features=X_dim + num_cond, out_features=hidden_dim))
        for i in range(num_layers)-1:
            self.fc_layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        '''

        self.fc1 = nn.Linear(in_features=X_dim + num_cond, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)

        # mean of latent space
        self.fc_mu = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        # deviation of latent space
        self.fc_logvar = nn.Linear(in_features=hidden_dim, out_features=latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)

        return x_mu, x_logvar

class Decoder(nn.Module):

    def __init__(self, X_dim, hidden_dim, latent_dim, num_cond):
        '''
        Decoder network with only fully connected layers and ReLU activations

        Args:
            X_dim: number of input variables (joint angles, ect.)
            hidden_dim: number of nodes of the fully connected layers
            latent_dims: number of nodes for additional variable z)
        '''

        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features=latent_dim + num_cond, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc4 = nn.Linear(in_features=hidden_dim, out_features=X_dim)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        # force input to be between [-1, 1]
        # force input to be between [-pi/4, pi/4]
        x = torch.acos(torch.zeros(1)).item() * 2 / 4 * torch.tanh(self.fc4(x))

        return x

class CVAE(nn.Module):

    def __init__(self, config):

        '''
        Conditional Autoencoder with fully connected encoder and decoder

        At the moment, only implemented for MNIST but will modified to fit for robot example

        For robot example, there is no bottleneck as we have 1 latent param z and 2 conditional params (x,y) to the 3
        input params

        The condition (x,y) is concatenated both with the input space X and the latent space z

        When computing the inverse kinematics, we draw a samples from the distribution of z. concatenate it with
        the observations (x,y) and feed the concatenated input into the decoder network.

        '''

        super(CVAE, self).__init__()

        self.latent_dim = config['latent_dim']
        self.X_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']

        self.num_condition = config['condition_dim']
        # self.preprocessor = Preprocessor()
        self.encoder = Encoder(self.X_dim, self.hidden_dim, self.latent_dim, self.num_condition)
        self.decoder = Decoder(self.X_dim, self.hidden_dim, self.latent_dim, self.num_condition)
        # self.postprocessor = Postprocessor()


    def forward(self, x, condition):

        # x = self.preprocessor(x)
        x = torch.cat((x, condition), dim=1)

        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        z = torch.cat((latent, condition), dim=1)
        x_recon = self.decoder(z)

        # x_recon = self.postprocessor(x_recon)

        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        '''
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
        '''
        # the reparameterization trick
        std = logvar.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()
        return eps.mul(std).add_(mu)


    def visualise_z(self, x, condition):

        with torch.no_grad():
            # x = self.preprocessor(x)
            x = torch.cat((x, condition), dim=1)
            latent_mu, latent_logvar = self.encoder(x)
            latent = self.latent_sample(latent_mu, latent_logvar)

        return latent


    def predict(self, tcp, device):

        # Sample z from standard normal distribution
        z = torch.randn(tcp.size()[0], self.latent_dim, device=device)
        x = torch.cat((z, tcp), dim=1)
        with torch.no_grad():
            x = self.decoder(x)
            # x = self.postprocessor(x)

        return x

    def save_checkpoint(self, epoch, optimizer, loss, PATH):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, PATH)

    def load_checkpoint(self, PATH, optimizer=None):
        checkpoint = torch.load(PATH)
        self.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        if not optimizer == None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return optimizer, epoch, loss
        else:
            return epoch, loss

    def save_weights(self, PATH):
        # TODO: save whole model state
        torch.save(self.state_dict(), PATH)

    def load_weights(self, PATH):
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(PATH))
        else:
            self.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))