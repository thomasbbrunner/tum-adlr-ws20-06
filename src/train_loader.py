import torch

from losses import VAE_loss_ROBOT_SIM

from robotsim_dataset import RobotSimDataset
import robotsim

import matplotlib.pyplot as plt
from utils import *
from losses import *

########################################################################################################################
# TRAIN METHOD FOR CVAE
########################################################################################################################

def train_CVAE(model, config, dataloader, device):

    # set to training mode
    model.train()

    train_loss_avg = []
    recon_loss_avg = []
    kl_loss_avg = []

    num_trainable_parameters = sum(p.numel() for p in model.parameters())

    print('TRAINABLE PARAMETERS: ', num_trainable_parameters)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr_rate'],
                                 weight_decay=config['weight_decay'])

    # define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config['step_size'],
                                                gamma=config['gamma'])

    for epoch in range(config['num_epochs']):

        train_loss_avg.append(0)
        recon_loss_avg.append(0)
        kl_loss_avg.append(0)
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

            # compute losses
            recon_loss = F.mse_loss(image_batch_recon, joint_batch, reduction='sum')
            kldivergence = -0.5 * torch.sum(1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())

            loss = recon_loss + config['variational_beta'] * kldivergence

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # one step of the optimizer
            optimizer.step()

            train_loss_avg[-1] += loss.item()
            recon_loss_avg[-1] += recon_loss.item()
            kl_loss_avg[-1] += kldivergence.item()
            num_batches += 1

        # perform step of lr-scheduler
        scheduler.step()

        if epoch > 1 and epoch % config['checkpoint_epoch'] == 0:
            model.save_checkpoint(epoch=epoch, optimizer=optimizer, loss=loss,
                                  PATH=config['checkpoint_dir'] + 'CVAE_' + str(config['dof']) + '_epoch_' + str(epoch))

        train_loss_avg[-1] /= num_batches
        recon_loss_avg[-1] /= num_batches
        kl_loss_avg[-1] /= num_batches

        print('Epoch [%d / %d] average reconstruction error: %f, average kl error: %f, average overall error: %f'
              % (epoch + 1, config['num_epochs'], recon_loss_avg[-1], kl_loss_avg[-1], train_loss_avg[-1]))

    plt.title('AVG LOSS HISTORY')
    plt.xlabel('EPOCHS')
    plt.ylabel('AVG LOSS')
    plt.plot(train_loss_avg, '-b', label='Total loss')
    plt.plot(recon_loss_avg, '-r', label='MSE loss')
    plt.plot(kl_loss_avg, '-k', label='KL loss')
    plt.legend()
    plt.savefig('figures/avg_train_loss_CVAE_tmp_' + str(config['dof']) + '.png')

########################################################################################################################
# TRAIN METHOD FOR INN
########################################################################################################################

def train_INN(model, config, dataloader, device):

    # try to overfit on a single batch
    # x_orig, y_orig = next(iter(dataloader))

    # set to training mode
    model.train()

    train_loss_avg = []
    train_loss_Ly_avg = []
    train_loss_Lz_avg = []
    train_loss_Lx_avg = []
    train_loss_Lx_avg_unweighted = []
    train_loss_Lxy_avg = []

    # show trainable parameters
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    num_trainable_parameters = sum(p.numel() for p in model.parameters())

    print('TRAINABLE PARAMETERS: ', num_trainable_parameters)

    optimizer = torch.optim.Adam(params=trainable_parameters, lr=config['lr_rate'],
                                 weight_decay=config['weight_decay'])
    # define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=config['step_size'],
                                                gamma=config['gamma'])
    # Padding in case xdim < total dim or yz_dim < total_dim
    # compute possible padding for input
    diff = config['total_dim'] - config['input_dim']
    # compute possible padding for output
    pad = config['total_dim'] - config['output_dim'] - config['latent_dim']

    zeros_noise_scale = config['zeros_noise_scale']
    y_noise_scale = config['y_noise_scale']

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

        ################################################################################################################

        for x, y in dataloader:

            # x = x_orig.clone()
            # y = y_orig.clone()

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

            # TODO: Add noise to x as well
            x += y_noise_scale * torch.randn(config['batch_size'], config['input_dim'], dtype=torch.float,
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

            # shorten y and output for latent loss computation: (z, pad_yz)
            y_short = torch.cat((y[:, :config['latent_dim']], y[:, -config['output_dim']:]), dim=1)
            output_short = torch.cat((output[:, :config['latent_dim']], output[:, -config['output_dim']:].data), dim=1)

            L_y = config['weight_Ly'] * MSE(output[:, config['latent_dim']:], y[:, config['latent_dim']:], reduction='mean')
            # print('L_y: ', config['weight_Ly'] * L_y)
            L_z = config['weight_Lz'] * MMD(output_short, y_short, device)
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

            # TODO: What is the benefit of that loss?
            UNWEIGHTED_LOSS = MMD(output_inv_rand[:, :config['input_dim']], x[:, :config['input_dim']], device)
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

        ################################################################################################################

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

        # print('Unweighted x-MMD loss: %f' % train_loss_Lx_avg_unweighted[-1])

    plt.title('AVG LOSS HISTORY')
    plt.xlabel('EPOCHS')
    plt.ylabel('AVG LOSS')
    plt.plot(train_loss_avg, '-b', label='Total loss')
    plt.plot(train_loss_Ly_avg, '-r', label='y-MSE loss')
    plt.plot(train_loss_Lxy_avg, '-m', label='x-MSE loss')
    plt.plot(train_loss_Lz_avg, '-g', label='z-MMD loss')
    plt.plot(train_loss_Lx_avg, '-k', label='x-MMD loss')
    plt.legend()
    plt.savefig('figures/avg_train_loss_INN_' + str(config['dof']) + '.png')