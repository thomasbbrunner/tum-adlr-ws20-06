import torch

from robotsim_dataset import RobotSimDataset
import robotsim

import matplotlib.pyplot as plt
from utils import *
from losses import *

'''
Training pipelines for INN and CVAE
'''

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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config['milestones'],
                                                     gamma=config['gamma'])

    for epoch in range(config['num_epochs']):

        train_loss_avg.append(0)
        recon_loss_avg.append(0)
        kl_loss_avg.append(0)
        num_batches = 0

        for joint_batch, coord_batch in dataloader:

            joint_batch = joint_batch.to(device)
            coord_batch = coord_batch.to(device)

            # forward pass only accepts float
            joint_batch = joint_batch.float()
            coord_batch = coord_batch.float()

            # apply sine and cosine to joint angles
            joint_batch = preprocess(joint_batch, config=config)

            # forward propagation
            image_batch_recon, latent_mu, latent_logvar = model(joint_batch, coord_batch)

            recon_loss = MSEloss4joints(image_batch_recon, joint_batch, config=config)
            kldivergence = KL_divergence(latent_mu, latent_logvar)

            # adapt size of variational beta to epoch
            # if epoch < config['num_epochs'] / 2:
            #     beta = config['variational_beta']
            # elif epoch >= config['num_epochs'] / 2 and epoch < 3 * config['num_epochs'] / 4:
            #     beta = 10 * config['variational_beta']
            # else:
            #     beta = 100 * config['variational_beta']

            loss = recon_loss + config['variational_beta'] * kldivergence

            # backpropagation
            optimizer.zero_grad()
            loss.backward()

            # one step of the optimizer
            optimizer.step()

            train_loss_avg[-1] += loss.detach()
            recon_loss_avg[-1] += recon_loss.detach()
            kl_loss_avg[-1] += kldivergence.detach()
            num_batches += 1

        # perform step of lr-scheduler
        scheduler.step()

        # save checkpoint
        if epoch > 1 and epoch % config['checkpoint_epoch'] == 0:
            model.save_checkpoint(
                epoch=epoch, optimizer=optimizer, loss=loss,
                PATH='{}CVAE_{}DOF_epoch_{}'.format(config['checkpoint_dir'], config['dof'], epoch))
            print('CHECKPOINT SAVED.')

        train_loss_avg[-1] /= num_batches
        recon_loss_avg[-1] /= num_batches
        kl_loss_avg[-1] /= num_batches

        print('Epoch [%d / %d] avg reconstruction error: %f, avg kl error: %f, avg overall error: %f'
              % (epoch + 1, config['num_epochs'], recon_loss_avg[-1], kl_loss_avg[-1],
                 train_loss_avg[-1]))


    fig = plt.figure()
    plt.title('TOTAL AVG LOSS HISTORY')
    plt.xlabel('EPOCHS')
    plt.ylabel('AVG LOSS')
    plt.plot(train_loss_avg, '-b', label='Total loss')
    # plt.legend()
    plt.savefig('figures/total_avg_train_loss_CVAE_{}DOF.png'.format(config['dof']))

    fig = plt.figure()
    plt.title('AVG LOSS HISTORY FOR RECONSTRUCTION ERROR')
    plt.xlabel('EPOCHS')
    plt.ylabel('AVG LOSS')
    plt.plot(recon_loss_avg, '-r', label='MSE loss')
    # plt.legend()
    plt.savefig('figures/recon_avg_train_loss_CVAE_{}DOF.png'.format(config['dof']))

    fig = plt.figure()
    plt.title('AVG LOSS HISTORY FOR KL DIVERGENCE')
    plt.xlabel('EPOCHS')
    plt.ylabel('AVG LOSS')
    plt.plot(kl_loss_avg, '-k', label='KL loss')
    # plt.legend()
    plt.savefig('figures/kl_avg_train_loss_CVAE_{}DOF.png'.format(config['dof']))


########################################################################################################################
# TRAIN METHOD FOR INN
########################################################################################################################

def train_INN(model, config, dataloader, device):

    # for debugging
    torch.autograd.set_detect_anomaly(True)

    # try to overfit on a single batch
    # x_orig, y_orig = next(iter(dataloader))

    # set to training mode
    model.train()

    train_loss_avg = []
    train_loss_Ly_avg = []
    train_loss_Lz_avg = []
    train_loss_Lx_avg = []
    train_loss_Lxy_avg = []

    # show trainable parameters
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    num_trainable_parameters = sum(p.numel() for p in model.parameters())
    print('TRAINABLE PARAMETERS: ', num_trainable_parameters)

    # TODO: which eps is the best for stability ?
    optimizer = torch.optim.Adam(params=trainable_parameters, lr=config['lr_rate'], eps=1e-6,
                                 weight_decay=config['weight_decay'])
    # define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=config['milestones'],
                                                gamma=config['gamma'])

    # Padding in case x_dim < total dim
    X_PAD_DIM = config['total_dim'] - config['input_dim']
    # Padding in case y_dim + z_dim < total_dim
    Y_PAD_DIM = config['total_dim'] - config['output_dim'] - config['latent_dim']

    zeros_noise_scale = config['zeros_noise_scale']
    y_noise_scale = config['y_noise_scale']

    for epoch in range(config['num_epochs']):

        train_loss_avg.append(0)
        train_loss_Ly_avg.append(0)
        train_loss_Lz_avg.append(0)
        train_loss_Lx_avg.append(0)
        train_loss_Lxy_avg.append(0)
        num_batches = 0

        # If MMD on x-space is present from the start, the model can get stuck.
        # Instead, ramp it up exponentially.
        loss_factor = min(1., 2. * 0.002 ** (1. - (float(epoch) / config['num_epochs'])))
        # loss_factor = 1.0

        ################################################################################################################

        for x, y in dataloader:

            # x = x_orig.clone()
            # y = y_orig.clone()

            x, y = x.to(device), y.to(device)
            x = x.float()
            y = y.float()
            x = preprocess(x, config=config)
            # This is used later for training the inverse pass
            y_clean = y.clone()

            ############################################################################################################
            # FORWARD STEP
            ############################################################################################################

            # Insert noise
            X_PAD = zeros_noise_scale * torch.randn(config['batch_size'], X_PAD_DIM, device=device)
            Y_PAD = zeros_noise_scale * torch.randn(config['batch_size'], Y_PAD_DIM, device=device)

            y += y_noise_scale * torch.randn(config['batch_size'], config['output_dim'], dtype=torch.float,
                                             device=device)
            x += y_noise_scale * torch.randn(config['batch_size'], config['input_dim'], dtype=torch.float,
                                             device=device)

            # Sample z from standard normal distribution
            z = torch.randn(config['batch_size'], config['latent_dim'], device=device)

            # Concatenate
            x = torch.cat((x, X_PAD), dim=1)
            y = torch.cat((z, Y_PAD, y), dim=1)

            optimizer.zero_grad()

            # forward propagation
            output = model(x)

            # shorten y and output: (z, y)
            y_short = torch.cat((y[:, :config['latent_dim']], y[:, -config['output_dim']:]), dim=1)
            output_short = torch.cat((output[:, :config['latent_dim']], output[:, -config['output_dim']:].data), dim=1)

            L_y = config['weight_Ly'] * MSE(output[:, config['latent_dim']:], y[:, config['latent_dim']:],
                                            reduction='sum')
            L_z = config['weight_Lz'] * MMD(output_short, y_short, device)

            loss_forward = L_y + L_z
            loss = loss_forward.data.detach()

            # backpropagation
            # Do not free intermediate results in order to accumulate grads later from forward and backward
            loss_forward.backward(retain_graph=True)

            ############################################################################################################
            # BACKWARD STEP
            ############################################################################################################

            # Insert noise
            Y_PAD = zeros_noise_scale * torch.randn(config['batch_size'], Y_PAD_DIM, device=device)

            y = y_clean + y_noise_scale * torch.randn(config['batch_size'], config['output_dim'], dtype=torch.float,
                                             device=device)

            orig_z_perturbed = (output[:, :config['latent_dim']] + y_noise_scale *
                                torch.randn(config['batch_size'], config['latent_dim'], device=device))

            y_inv = torch.cat((orig_z_perturbed, Y_PAD, y), dim=1)

            y_inv_rand = torch.cat((torch.randn(config['batch_size'], config['latent_dim'], device=device), Y_PAD, y),
                                   dim=1)

            output_inv = model(y_inv, inverse=True)
            output_inv_rand = model(y_inv_rand, inverse=True)

            # forces padding dims to be ignored
            L_xy = config['weight_Lxy'] * MSEloss4joints(output_inv, x, config=config)

            L_x = config['weight_Lx'] * loss_factor * MMD(output_inv_rand[:, :config['input_dim']],
                                                          x[:, :config['input_dim']], device)

            loss_backward = L_xy + L_x
            loss += loss_backward.data.detach()

            loss_backward.backward()

            # very important such that grads in subnetworks dont explode!!!!!
            for p in model.parameters():
                p.grad.data.clamp_(-15.00, 15.00)

            # one step of the optimizer
            optimizer.step()

            train_loss_avg[-1] += loss

            train_loss_Ly_avg[-1] += L_y.data.detach()
            # if torch.isnan(train_loss_Ly_avg[-1]):
            #     raise Exception('NaN in train_loss_Ly_avg[-1] loss detected!')

            train_loss_Lz_avg[-1] += L_z.data.detach()
            # if torch.isnan(train_loss_Lz_avg[-1]):
            #     raise Exception('NaN in train_loss_Lz_avg[-1] loss detected!')

            train_loss_Lx_avg[-1] += L_x.data.detach()
            # if torch.isnan(train_loss_Lx_avg[-1]):
            #     raise Exception('NaN in Lx loss detected!')

            train_loss_Lxy_avg[-1] += L_xy.data.detach()
            # if torch.isnan(train_loss_Lxy_avg[-1]):
            #     raise Exception('NaN in Lxy loss detected!')

            num_batches += 1

        ################################################################################################################

        # perform step of lr-scheduler
        scheduler.step()

        # save checkpoint
        if epoch > 1 and epoch % config['checkpoint_epoch'] == 0:
            model.save_checkpoint(
                epoch=epoch, optimizer=optimizer, loss=loss,
                PATH='{}INN_{}DOF_epoch_{}'.format(config['checkpoint_dir'], config['dof'], epoch))
            print('CHECKPOINT SAVED.')

        train_loss_avg[-1] /= num_batches
        train_loss_Ly_avg[-1] /= num_batches
        train_loss_Lz_avg[-1] /= num_batches
        train_loss_Lx_avg[-1] /= num_batches
        train_loss_Lxy_avg[-1] /= num_batches

        print('Epoch [%d / %d] weighted avg y-MSE loss: %f, weighted avg y-MMD loss: %f, weighted avg x-MMD loss: %f, '
              'weighted avg x-MSE loss: %f, Overall avg loss: %f'
              % (epoch + 1, config['num_epochs'], train_loss_Ly_avg[-1],
                                          train_loss_Lz_avg[-1], train_loss_Lx_avg[-1], train_loss_Lxy_avg[-1], train_loss_avg[-1]))

    plt.title('AVG LOSS HISTORY')
    plt.xlabel('EPOCHS')
    plt.ylabel('AVG LOSS')
    plt.plot(train_loss_avg, '-b', label='Total loss')
    plt.plot(train_loss_Ly_avg, '-r', label='y-MSE loss')
    plt.plot(train_loss_Lxy_avg, '-m', label='x-MSE loss')
    plt.plot(train_loss_Lz_avg, '-g', label='z-MMD loss')
    plt.plot(train_loss_Lx_avg, '-k', label='x-MMD loss')
    plt.legend()
    plt.savefig('figures/avg_train_loss_INN_{}DOF.png'.format(config['dof']))