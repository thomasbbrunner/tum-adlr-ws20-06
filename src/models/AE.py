import torch
from torch import nn

# 3DoF manipulator in 2D

class AE(nn.Module):


    def __init__(self):
        super(AE, self).__init__()

        # TODO: Implement encoder and decoder

        self.encoder = None
        self.decoder = None

    def forward(self, X):

        # TODO: Implement forward method

        return X