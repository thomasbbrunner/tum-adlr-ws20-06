import torch

from torch.utils.data import DataLoader

from models.CVAE import *
from losses import VAE_loss_ROBOT_SIM

from robotsim_dataset import RobotSimDataset
import robotsim

import matplotlib.pyplot as plt
from utils import *


if __name__ == '__main__':

    '''
    Trains a conditional autoencoder on the robotsim dataset
    '''

    ####################################################################################################################
    # TO MODIFY
    ####################################################################################################################

    X_dim = 3 * 2 # number of robot links
    hidden_dim = 50 # number of neurons per fully connected layer
    latent_dim = 2
    num_cond = 2 # (x, y) coordinates of end-effector
    num_epochs = 10
    batch_size = 500
    learning_rate = 1e-3
    weight_decay = 1e-4 # 1.6e-5
    variational_beta = 1 / 4
    use_gpu = False
    PATH = 'weights/ROBOTSIM_CVAE_SIN_COS'

    ####################################################################################################################
    # LOAD DATASET
    ####################################################################################################################

    robot = robotsim.RobotSim2D(3, [3, 3, 3])

    # INPUT: 3 joint angles
    # OUTPUT: (x,y) coordinate of end-effector
    # dataset = RobotSimDataset(robot, 100)
    dataset = RobotSimDataset(robot, 100)

    # train test split
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [700000, 300000])

    # print('LENGTH OF DATASET: ', train_dataset.__len__())
    # print('ITEM 5 OF DATASET: ', train_dataset.__getitem__(5))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ####################################################################################################################
    # BUILD MODEL
    ####################################################################################################################

    # TODO: Implement learning rate decay
    # TODO: Implement checkpoints
    # TODO: Change prior distribution
    # TODO: Consider joint limits
    # TODO: Perform Random Search for Hyperparameters

    cvae = CVAE(X_dim, hidden_dim, latent_dim, num_cond, classification=False)

    device = torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")
    cvae = cvae.to(device)

    optimizer = torch.optim.Adam(params=cvae.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ####################################################################################################################
    # TRAINING
    ####################################################################################################################

    # set to training mode
    cvae.train()

    train_loss_avg = []

    print('Training ...')
    for epoch in range(num_epochs):

        train_loss_avg.append(0)
        num_batches = 0

        for joint_batch, coord_batch in train_dataloader:

            joint_batch = joint_batch.to(device)

            # forward pass only accepts float
            joint_batch = joint_batch.float()
            coord_batch = coord_batch.float()

            # apply sine and cosine to joint angles
            joint_batch = preprocess(joint_batch)

            # forward propagation
            # print('FORWARD PASS ...')
            image_batch_recon, latent_mu, latent_logvar = cvae(joint_batch, coord_batch)

            # print('joint_batch: ', joint_batch[0, :])
            # print('image_batch_recon: ', image_batch_recon[0, :])

            # print('COMPUTE LOSS ...')
            # reconstruction and KL loss
            loss = VAE_loss_ROBOT_SIM(image_batch_recon, joint_batch, latent_mu, latent_logvar, variational_beta)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            # print('BACKWARD PASS ...')

            # one step of the optimizer
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1

        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch + 1, num_epochs, train_loss_avg[-1]))

    cvae.save_weights(PATH)

    fig = plt.figure()
    plt.title('AVG LOSS HISTORY')
    plt.xlabel('EPOCHS')
    plt.ylabel('AVG LOSS')
    plt.plot(train_loss_avg)
    plt.savefig('avg_train_loss_SIN_COS.png')