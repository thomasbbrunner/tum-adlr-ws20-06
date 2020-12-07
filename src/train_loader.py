import torch

from losses import VAE_loss_ROBOT_SIM

from robotsim_dataset import RobotSimDataset
import robotsim

import matplotlib.pyplot as plt
from utils import *
from losses import *

def train_CVAE(model, config, dataloader, device):

    # set to training mode
    model.train()

    train_loss_avg = []

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr_rate'],
                                 weight_decay=config['weight_decay'])

    # define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config['step_size'],
                                                gamma=config['gamma'])

    for epoch in range(config['num_epochs']):

        train_loss_avg.append(0)
        num_batches = 0

        for joint_batch, coord_batch in dataloader:

            joint_batch = joint_batch.to(device)

            # forward pass only accepts float
            joint_batch = joint_batch.float()
            coord_batch = coord_batch.float()

            # apply sine and cosine to joint angles
            joint_batch = preprocess(joint_batch)

            # forward propagation
            image_batch_recon, latent_mu, latent_logvar = model(joint_batch, coord_batch)

            # reconstruction and KL loss
            loss = VAE_loss_ROBOT_SIM(image_batch_recon, joint_batch, latent_mu, latent_logvar,
                                      config['variational_beta'])

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # one step of the optimizer
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1

        # perform step of lr-scheduler
        scheduler.step()

        if epoch > 1 and epoch % config['checkpoint_epoch'] == 0:
            model.save_checkpoint(epoch=epoch, optimizer=optimizer, loss=loss,
                                  PATH=config['checkpoint_dir'] + 'CVAE_' + str(config['dof']) + '_epoch_' + str(epoch))

        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch + 1, config['num_epochs'], train_loss_avg[-1]))

    return train_loss_avg

def train_INN(model, config, dataloader, device):

    # set to training mode
    model.train()

    train_loss_avg = []

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr_rate'],
                                 weight_decay=config['weight_decay'])

    # define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config['step_size'],
                                                gamma=config['gamma'])

    for epoch in range(config['num_epochs']):

        train_loss_avg.append(0)
        num_batches = 0

        for x, y in dataloader:

            x, y = x.to(device), y.to(device)

            # forward pass only accepts float
            x = x.float()
            y = y.float()

            # apply sine and cosine to joint angles
            x = preprocess(x)

            # Sample z from standard normal distribution
            z_dim = config['input_dim'] - config['output_dim']
            z = torch.randn(y.size()[0], z_dim, device=device)

            # Concatenate y and z
            y = torch.cat((z, y), dim=1)

            optimizer.zero_grad()

            # forward propagation
            output = model(x)

            # Compute loss
            loss = INN_loss_ROBOT_SIM(output, y, config, device)

            # backpropagation
            loss.backward()

            # one step of the optimizer
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            num_batches += 1

        # perform step of lr-scheduler
        scheduler.step()

        if epoch > 1 and epoch % config['checkpoint_epoch'] == 0:
            model.save_checkpoint(epoch=epoch, optimizer=optimizer, loss=loss,
                                  PATH=config['checkpoint_dir'] + 'INN_' + str(config['dof']) + '_epoch_' + str(epoch))

        train_loss_avg[-1] /= num_batches
        print('Epoch [%d / %d] average reconstruction error: %f' % (epoch + 1, config['num_epochs'], train_loss_avg[-1]))

    return train_loss_avg