import torch
from torch import nn
import torch.nn.functional as F
from utils import onehot

'''
 Sources
 -------
 https://github.com/VLL-HD/FrEIA/blob/master/FrEIA/
 '''

class sub_network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(sub_network, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc3 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.f1(x))
        x = F.leaky_relu(self.f2(x))
        x = F.leaky_relu(self.f3(x))
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

        # Perform forward kinematics
        if not inverse:
            pass

        # Perform inverse kinematics
        else:
            pass

class INN(nn.Module):

    def __init__(self, input_dim, condition_dim, hidden_dim, latent_dim):

        super(INN, self).__init__()

    def forward(self, x):
        return x

    def inverse(self, x):
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
        self.load_state_dict(torch.load(PATH))