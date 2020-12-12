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
    train_loss_Ly_avg = []
    train_loss_Lz_avg = []
    train_loss_Lx_avg = []
    train_loss_Lx_avg_unweighted = []
    train_loss_Lxy_avg = []

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr_rate'],
                                 weight_decay=config['weight_decay'])
    # define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config['step_size'],
                                                gamma=config['gamma'])
    # Padding in case xdim < total dim or yz_dim < total_dim
    # compute possible padding for input
    diff = config['total_dim'] - config['input_dim']
    # compute possible padding for output
    pad = config['total_dim'] - config['output_dim'] - config['latent_dim']

    zeros_noise_scale = 5e-2
    y_noise_scale = 1e-1

    for epoch in range(config['num_epochs']):

        train_loss_avg.append(0)
        train_loss_Ly_avg.append(0)
        train_loss_Lz_avg.append(0)
        train_loss_Lx_avg.append(0)
        train_loss_Lxy_avg.append(0)
        train_loss_Lx_avg_unweighted.append(0)
        num_batches = 0

        # If MMD on x-space is present from the start, the model can get stuck.
        # Instead, ramp it up exponentially.
        # loss_factor = min(1., 2. * 0.002 ** (1. - (float(epoch) / config['num_epochs'])))
        loss_factor = 1.0

        for x, y in dataloader:

            x, y = x.to(device), y.to(device)
            # forward pass only accepts float
            x = x.float()
            y = y.float()
            # apply sine and cosine to joint angles
            x = preprocess(x)
            # This is used later for training the inverse pass
            y_clean = y.clone()

            ############################################################################################################
            # FORWARD STEP
            ############################################################################################################

            # Insert noise
            pad_x = zeros_noise_scale * torch.randn(config['batch_size'], diff, device=device)
            pad_yz = zeros_noise_scale * torch.randn(config['batch_size'], pad, device=device)
            y += y_noise_scale * torch.randn(config['batch_size'], config['output_dim'], dtype=torch.float,
                                             device=device)
            # Sample z from standard normal distribution
            z = torch.randn(config['batch_size'], config['latent_dim'], device=device)

            # Concatenate
            x = torch.cat((x, pad_x), dim=1)
            y = torch.cat((z, pad_yz, y), dim=1)

            ############################################################################################################

            optimizer.zero_grad()

            # forward propagation
            output = model(x)

            # shorten y and output for latent loss computation
            y_short = torch.cat((y[:, :config['latent_dim']], y[:, -config['output_dim']:]), dim=1)
            output_short = torch.cat((output[:, :config['latent_dim']], output[:, -config['output_dim']:].data), dim=1)

            L_y = config['weight_Ly'] * MSE(output[:, config['latent_dim']:], y[:, config['latent_dim']:], reduction='mean')
            # print('L_y: ', config['weight_Ly'] * L_y)
            L_z = config['weight_Lz'] * MMD(output_short, y_short)
            # print('L_z: ', config['weight_Lz'] * L_z)
            loss_forward = L_y + L_z
            loss = loss_forward.data.item()

            # backpropagation
            # Do not free intermediate results in order to accumulate grads later from forward and backward
            loss_forward.backward(retain_graph=True)

            ############################################################################################################
            # BACKWARD STEP
            ############################################################################################################

            # Insert noise
            pad_yz = zeros_noise_scale * torch.randn(config['batch_size'], pad, device=device)
            y = y_clean + y_noise_scale * torch.randn(config['batch_size'], config['output_dim'], dtype=torch.float,
                                                      device=device)
            orig_z_perturbed = (output[:, :config['latent_dim']] + y_noise_scale *
                                torch.randn(config['batch_size'], config['latent_dim'], device=device))
            y_inv = torch.cat((orig_z_perturbed, pad_yz, y), dim=1)
            y_inv_rand = torch.cat((torch.randn(config['batch_size'], config['latent_dim'], device=device), pad_yz, y),
                                   dim=1)

            ############################################################################################################

            output_inv = model(y_inv, inverse=True)
            output_inv_rand = model(y_inv_rand, inverse=True)

            # forces padding dims to be ignored
            L_xy = config['weight_Lxy'] * MSE(output_inv, x, reduction='mean')
            # print('L_xy: ', config['weight_Ly'] * L_xy)

            UNWEIGHTED_LOSS = MMD(output_inv_rand[:, :config['input_dim']], x[:, :config['input_dim']])
            L_x = config['weight_Lx'] * loss_factor * UNWEIGHTED_LOSS
            # print('L_x: ', config['weight_Lx'] * L_x)

            loss_backward = L_x + L_xy
            loss += loss_backward.data.item()
            loss_backward.backward()

            for p in model.parameters():
                p.grad.data.clamp_(-15.00, 15.00)

            # one step of the optimizer
            optimizer.step()

            train_loss_avg[-1] += loss
            train_loss_Ly_avg[-1] += L_y.data.item()
            train_loss_Lz_avg[-1] += L_z.data.item()
            train_loss_Lx_avg[-1] += L_x.data.item()
            train_loss_Lxy_avg[-1] += L_xy.data.item()
            train_loss_Lx_avg_unweighted[-1] += UNWEIGHTED_LOSS.data.item()

            num_batches += 1

        # perform step of lr-scheduler
        scheduler.step()

        if epoch > 1 and epoch % config['checkpoint_epoch'] == 0:
            model.save_checkpoint(epoch=epoch, optimizer=optimizer, loss=loss,
                                  PATH=config['checkpoint_dir'] + 'INN_' + str(config['dof']) + '_epoch_' + str(epoch))

        train_loss_avg[-1] /= num_batches
        train_loss_Ly_avg[-1] /= num_batches
        train_loss_Lz_avg[-1] /= num_batches
        train_loss_Lx_avg[-1] /= num_batches
        train_loss_Lxy_avg[-1] /= num_batches
        train_loss_Lx_avg_unweighted[-1] /= num_batches

        print('Epoch [%d / %d] weighted average y-MSE loss: %f, weighted average y-MMD loss: %f, '
              'weighted average x-MSE loss: %f, weighted average x-MMD loss: %f, Overall average loss: %f'
              % (epoch + 1, config['num_epochs'], train_loss_Ly_avg[-1],
                                          train_loss_Lz_avg[-1], train_loss_Lxy_avg[-1], train_loss_Lx_avg[-1], train_loss_avg[-1]))

        print('Unweighted x-MMD loss: %f' % train_loss_Lx_avg_unweighted[-1])

    return train_loss_avg