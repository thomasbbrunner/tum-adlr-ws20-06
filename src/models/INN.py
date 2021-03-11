import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class FixedRandomPermutation(nn.Module):
    """Layer with random but fixed permutations in order to mix the data"""

    def __init__(self, input_dim, seed):
        super(FixedRandomPermutation, self).__init__()

        np.random.seed(seed)
        self.in_channels = input_dim
        self.permutation = np.random.permutation(self.in_channels)
        np.random.seed()
        self.permutation_inv = np.zeros_like(self.permutation)

        for i, p in enumerate(self.permutation):
            self.permutation_inv[p] = i
        if torch.cuda.is_available():
            self.permutation = torch.cuda.LongTensor(self.permutation)
            self.permutation_inv = torch.cuda.LongTensor(self.permutation_inv)
        else:
            self.permutation = torch.LongTensor(self.permutation)
            self.permutation_inv = torch.LongTensor(self.permutation_inv)

    def forward(self, x, inverse=False):
        if not inverse:
            x = x[:, self.permutation]
        else:
            x = x[:, self.permutation_inv]
        return x

class sub_network(nn.Module):
    """Fully connected subnetwork of a single coupling block"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(sub_network, self).__init__()

        # create list of hidden layers
        self.fcs = nn.ModuleList([nn.Linear(in_features=input_dim, out_features=hidden_dim)])
        self.fcs.extend([nn.Linear(in_features=hidden_dim, out_features=hidden_dim) for i in range(1, num_layers-1)])
        self.fcs.append(nn.Linear(in_features=hidden_dim, out_features=output_dim))

    def forward(self, x):
        for layer in self.fcs:
            x = F.leaky_relu(layer(x))
        return x

class AffineCouplingBlock(nn.Module):
    """Coupling layer with affine transformations and 4 single subnetworks for each learned coefficient

    Refer to implementation: https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/

    """
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(AffineCouplingBlock, self).__init__()

        # Split input U into two halves
        self.u1_dim = input_dim // 2
        self.u2_dim = input_dim - input_dim // 2
        self.clamp = 5.0 # important to bound exp

        # Define scale and translation subnetworks for the two complementary affine coupling layers
        self.s1 = sub_network(self.u1_dim, hidden_dim, self.u2_dim, num_layers)
        self.t1 = sub_network(self.u1_dim, hidden_dim, self.u2_dim, num_layers)
        self.s2 = sub_network(self.u2_dim, hidden_dim, self.u1_dim, num_layers)
        self.t2 = sub_network(self.u2_dim, hidden_dim, self.u1_dim, num_layers)

    def e(self, x):
        """Exponential function with clipped values to avoid too big values

        Args:
            x: input

        Returns: exponential with clipped values

        """
        return torch.exp(self.clamp * 0.636 * torch.atan(x))

    def forward(self, x, inverse=False):

        # Split x in two halves
        u1 = torch.narrow(x, 1, 0, self.u1_dim)
        u2 = torch.narrow(x, 1, self.u1_dim, self.u2_dim)

        # Perform forward kinematics
        if not inverse:
            # v1 = u1 dotprod exp(s2(u2)) + t2(u2)
            exp_2 = self.e(self.s2(u2))
            v1 = u1 * exp_2 + self.t2(u2)
            # v2 = u2 dotprod exp(s1(v1)) + t1(v1)
            exp_1 = self.e(self.s1(v1))
            v2 = u2 * exp_1 + self.t1(v1)

        # Perform inverse kinematics (names of u and v are swapped)
        else:
            # u2 = (v2-t1(v1)) dotprod exp(-s1(v1))
            exp_1 = self.e(-self.s1(u1))
            v2 = (u2 - self.t1(u1)) * exp_1
            # u1 = (v1-t2(u2)) dotprod exp(-s2(u2))
            exp_2 = self.e(-self.s2(v2))
            v1 = (u1 - self.t2(v2)) * exp_2

        return torch.cat((v1, v2), 1)


class INN(nn.Module):
    """Invertible Neural Network (INN)

    Paper: Analyzing inverse problems with invertible neural networks [L. Ardizzone et al. 2018]

    Implementation inspired from:
    https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/
    https://github.com/VLL-HD/analyzing_inverse_problems

    """

    def __init__(self, config):

        super(INN, self).__init__()
        self.total_dim = config['total_dim']
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.hidden_dim = config['hidden_dim']
        self.latent_dim = config['latent_dim']
        self.num_layers_subnet = config['num_layers_subnet']
        self.num_coupling_layers = config['num_coupling_layers']

        # create list of hidden layers
        self.fcs = nn.ModuleList()
        for i in range(self.num_coupling_layers-1):
            self.fcs.append(AffineCouplingBlock(self.total_dim, self.hidden_dim, self.num_layers_subnet))
            self.fcs.append(FixedRandomPermutation(self.total_dim, i))
        self.fcs.append(AffineCouplingBlock(self.total_dim, self.hidden_dim, self.num_layers_subnet))


    def forward(self, x, inverse=False):
        if not inverse:
            for layer in self.fcs:
                x = layer(x, inverse)
        else:
            for layer in reversed(self.fcs):
                x = layer(x, inverse)
        return x

    def predict(self, tcp, device):
        """Predicts joint angles dependent on the tcp + by sampling from N(0, 1)

        Args:
            tcp: (x, y) coordinates of end-effector
            device: 'cpu' or 'gpu'

        Returns: predicted joint angles

        """

        # Sample z from standard normal distribution
        z = torch.randn(tcp.size()[0], self.latent_dim, device=device)
        # Padding in case y_dim + z_dim < total_dim
        Y_PAD = torch.zeros(tcp.size()[0], self.total_dim - self.output_dim - self.latent_dim, device=device)
        # Perform inverse kinematics
        y_inv = torch.cat((z, Y_PAD, tcp), dim=1)
        with torch.no_grad():
            output_inv = self.forward(y_inv, inverse=True)
        return output_inv

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
