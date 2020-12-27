import torch
from torch import nn
import torch.nn.functional as F
from utils import onehot
import numpy as np
import matplotlib.pyplot as plt

'''

 Sources
 -------
 https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/
 
 https://github.com/VLL-HD/analyzing_inverse_problems
 
 '''

# Permutes input vector in a random but fixed way
class FixedRandomPermutation(nn.Module):
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

    def jacobian(self):
        pass

class sub_network(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(sub_network, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

class AffineCouplingBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AffineCouplingBlock, self).__init__()

        # Split input U into two halves
        self.u1_dim = input_dim // 2
        self.u2_dim = input_dim - input_dim // 2

        # Define scale and translation subnetworks for the two complementary affine coupling layers
        self.s1 = sub_network(self.u1_dim, hidden_dim, self.u2_dim)
        self.t1 = sub_network(self.u1_dim, hidden_dim, self.u2_dim)
        self.s2 = sub_network(self.u2_dim, hidden_dim, self.u1_dim)
        self.t2 = sub_network(self.u2_dim, hidden_dim, self.u1_dim)

    def forward(self, x, inverse=False):

        # Split x in two halves
        u1 = torch.narrow(x, 1, 0, self.u1_dim)
        u2 = torch.narrow(x, 1, self.u1_dim, self.u2_dim)

        # Perform forward kinematics
        if not inverse:
            # v1 = u1 dotprod exp(s2(u2)) + t2(u2)
            exp_2 = torch.exp(self.s2(u2))
            v1 = u1 * exp_2 + self.t2(u2)

            # v2 = u2 dotprod exp(s1(v1)) + t1(v1)
            exp_1 = torch.exp(self.s1(v1))
            v2 = u2 * exp_1 + self.t1(v1)


        # Perform inverse kinematics (names of u and v are swapped)
        else:
            # u2 = (v2-t1(v1)) dotprod exp(-s1(v1))
            exp_1 = torch.exp(-self.s1(u1))
            v2 = (u2 - self.t1(u1)) * exp_1

            # u1 = (v1-t2(u2)) dotprod exp(-s2(u2))
            exp_2 = torch.exp(-self.s2(v2))
            v1 = (u1 - self.t2(v2)) * exp_2

        return torch.cat((v1, v2), 1)

    def jacobian(self, inverse=False):
        pass

class INN(nn.Module):

    def __init__(self, config):

        super(INN, self).__init__()
        self.total_dim = config['total_dim']
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.hidden_dim = config['hidden_dim']
        self.latent_dim = config['latent_dim']
        self.batch_size = config['batch_size']
        self.y_noise_scale = config['y_noise_scale']
        self.zeros_noise_scale = config['zeros_noise_scale']

        self.block1 = AffineCouplingBlock(self.input_dim, self.hidden_dim)
        self.perm1 = FixedRandomPermutation(self.input_dim, 1)
        self.block2 = AffineCouplingBlock(self.input_dim, self.hidden_dim)

        self.perm2 = FixedRandomPermutation(self.input_dim, 2)
        self.block3 = AffineCouplingBlock(self.input_dim, self.hidden_dim)
        self.perm3 = FixedRandomPermutation(self.input_dim, 3)
        self.block4 = AffineCouplingBlock(self.input_dim, self.hidden_dim)
        # self.perm4 = FixedRandomPermutation(self.input_dim, 4)
        # self.block5 = AffineCouplingBlock(self.input_dim, self.hidden_dim)
        # self.perm5 = FixedRandomPermutation(self.input_dim, 5)
        # self.block6 = AffineCouplingBlock(self.input_dim, self.hidden_dim)


    def forward(self, x, inverse=False):

        if not inverse:
            # coupling layer
            x = self.block1(x, inverse)
            # shuffle components (fixed permutation)
            # ensures that every single input variable affects every single output variable
            x = self.block2(self.perm1(x, inverse), inverse)
            x = self.block3(self.perm2(x, inverse), inverse)
            x = self.block4(self.perm3(x, inverse), inverse)
            # x = self.block5(self.perm4(x, inverse), inverse)
            # x = self.block6(self.perm5(x, inverse), inverse)


        else:
            x = self.block4(x, inverse)
            # x = self.block6(x, inverse)
            # x = self.block5(self.perm5(x, inverse), inverse)
            # x = self.block4(self.perm4(x, inverse), inverse)
            x = self.block3(self.perm3(x, inverse), inverse)
            x = self.block2(self.perm2(x, inverse), inverse)
            x = self.block1(self.perm1(x, inverse), inverse)

        return x

    def predict(self, tcp, device):

        # Sample z from standard normal distribution
        z = torch.randn(tcp.size()[0], self.latent_dim, device=device)

        # Padding in case yz_dim < total_dim
        # pad_yz = self.zeros_noise_scale * torch.randn(self.batch_size, self.total_dim -
        #                                          self.input_dim - self.latent_dim, device=device)
        pad_yz = torch.zeros(tcp.size()[0], self.total_dim - self.output_dim - self.latent_dim, device=device)

        # Perform inverse kinematics
        y_inv = torch.cat((z, pad_yz, tcp), dim=1)
        with torch.no_grad():
            output_inv = self.forward(y_inv, inverse=True)

        return output_inv

    def visualise_z(self, config, x):

        assert x.size()[1] == config['total_dim']

        with torch.no_grad():
            y = self.forward(x, inverse=False)

        z = y[:, :config['latent_dim']]

        if z.size()[1] == 1:
            raise Exception('Not implemented yet')
        elif z.size()[1] == 2:
            fig = plt.figure()
            plt.title('Latent space')
            plt.xlabel('Z1')
            plt.ylabel('Z2')
            plt.scatter(z[:, 0], z[:, 1], c='g')
            plt.savefig('figures/Latent_space_INN_' + str(config['dof']) + '.png')
        else:
            # Perform principal component analysis to project z int 2D space
            U, S, V = torch.pca_lowrank(z, center=False)
            z_projected = torch.matmul(z, V[:, :2])
            fig = plt.figure()
            plt.title('Latent space')
            plt.xlabel('Z1')
            plt.ylabel('Z2')
            plt.scatter(z_projected[:, 0], z_projected[:, 1], c='g')
            plt.savefig('figures/Projected_latent_space_INN_' + str(config['dof']) + '.png')

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



# Testing example
if __name__ == '__main__':

    # inn = INN(input_dim=6, hidden_dim=128, output_dim=6)
    permutation = FixedRandomPermutation(input_dim=6, seed=42)

    print(permutation.in_channels)
    print(permutation.permutation)
    print(permutation.permutation_inv)

    x = torch.Tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])

    print('Original x: ', x)
    x_permuted = permutation(x, inverse=False)
    print('Permuted x: ', x_permuted)
    x_repermuted = permutation(x_permuted, inverse=True)
    print('x_repermuted x: ', x_repermuted)
