import torch
from torch import nn
import torch.nn.functional as F
from utils import onehot

'''
 Sources
 -------
 CVAE paper:  Learning structured output representation using deep conditional generative models [Sohn et al 2015]
 Code: https://github.com/graviraja/pytorch-sample-codes/blob/master/conditional_vae.py
 '''

class Encoder(nn.Module):
    def __init__(self, X_dim, hidden_dim, latent_dim, num_cond, classification=True):

        '''
        Encoder network with only fully connected layers and ReLU activations

        Args:
            X_dim: number of input variables (joint angles, ect.)
            hidden_dim: number of nodes of the fully connected layers
            latent_dims: number of nodes for additional variable z)
            num_classes: For classification in MNIST: 10 classes, for 2D robot: observations (x,y)
        '''

        super(Encoder, self).__init__()
        self.classification = classification

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

    def __init__(self, X_dim, hidden_dim, latent_dim, num_cond, classification=True):
        '''
        Decoder network with only fully connected layers and ReLU activations

        Args:
            X_dim: number of input variables (joint angles, ect.)
            hidden_dim: number of nodes of the fully connected layers
            latent_dims: number of nodes for additional variable z)
            num_classes: For classification in MNIST: 10 classes, for 2D robot: observations (x,y)
        '''

        super(Decoder, self).__init__()
        self.classification = classification
        self.fc1 = nn.Linear(in_features=latent_dim + num_cond, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc4 = nn.Linear(in_features=hidden_dim, out_features=X_dim)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        if self.classification:
            x = torch.sigmoid(self.fc4(x))
        else:
            x = self.fc4(x)
        return x

class CVAE(nn.Module):

    def __init__(self, X_dim, hidden_dim, latent_dim, num_cond, classification=True):

        '''
        Conditional Autoencoder with fully connected encoder and decoder

        At the moment, only implemented for MNIST but will modified to fit for robot example

        For robot example, there is no bottleneck as we have 1 latent param z and 2 conditional params (x,y) to the 3
        input params

        The condition (x,y) is concatenated both with the input space X and the latent space z

        When computing the inverse kinematics, we draw a samples from the distribution of z. concatenate it with
        the observations (x,y) and feed the concatenated input into the decoder network.

        Args:
            X_dim: number of input variables (joint angles, ect.)
            hidden_dim: number of nodes of the fully connected layers
            latent_dims: number of nodes for additional variable z)
            num_classes: For classification in MNIST: 10 classes, for 2D robot: observations (x,y)
        '''

        super(CVAE, self).__init__()

        self.classification = classification
        self.num_condition = num_cond
        self.encoder = Encoder(X_dim, hidden_dim, latent_dim, num_cond, classification)
        self.decoder = Decoder(X_dim, hidden_dim, latent_dim, num_cond, classification)


    def forward(self, x, condition):

        if self.classification:
            condition = onehot(condition.view(-1, 1), self.num_condition)

        x = torch.cat((x, condition), dim=1)

        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        z = torch.cat((latent, condition), dim=1)
        x_recon = self.decoder(z)

        return x_recon, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu


    def predict(self, x, condition):
        x = torch.cat((x, condition), dim=1)
        x = self.decoder(x)
        return x

    def save_weights(self, PATH):
        # TODO: save whole model state
        torch.save(self.state_dict(), PATH)

    def load_weights(self, PATH):
        self.load_state_dict(torch.load(PATH))