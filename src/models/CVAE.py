import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, X_dim, hidden_dim, latent_dim, num_classes):

        '''
        Args:
            input_dim:
            hidden_dim:
            latent_dims:
            num_classes:
        '''

        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(in_features=X_dim + num_classes, out_features=hidden_dim)
        self.fc_mu = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_dim, out_features=latent_dim)

    def forward(self, x):

        # x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)

        return x_mu, x_logvar

class Decoder(nn.Module):

    def __init__(self, X_dim, hidden_dim, latent_dim, num_classes):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_features=latent_dim + num_classes, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=X_dim)


    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        # x = x.view(x.size(0), 28, 28, 1)
        return x

class CVAE(nn.Module):

    def __init__(self, X_dim, hidden_dim, latent_dim, num_classes):
        super(CVAE, self).__init__()
        self.encoder = Encoder( X_dim, hidden_dim, latent_dim, num_classes)
        self.decoder = Decoder( X_dim, hidden_dim, latent_dim, num_classes)

    def forward(self, x, condition):

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

    def save_weights(self, PATH):
        torch.save(self.state_dict(), PATH)

    def load_weights(self, PATH):
        self.load_state_dict(torch.load(PATH))