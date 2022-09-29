from os import getpid
from os.path import join
from time import time
import numpy as np
import torch
import torch.optim as optim
from general_loss import CharbMSEWT, CharbDirect, CharbTV, create_vgg_model, vggfeature_loss
from torch.autograd import Variable


def log(x):
    return torch.log(x + 1e-10)


def train_epoch(epoch, args, D, G, vgg_models, data_loader, D_optimizer, G_optimizer):
    def reset_grad():
        D_optimizer.zero_grad()
        G_optimizer.zero_grad()

    D.train()
    G.train()
    pid = getpid()

    train_loss_d = 0.0
    step_loss_d = 0.0
    train_loss_g = 0.0
    step_loss_g = 0.0

    train_loss_char = 0.0
    train_loss_feat = 0.0
    train_loss_gan = 0.0

    data_idx = 0

    for batch_idx, (Z, X) in enumerate(data_loader):

        # Sample Z, X
        if args.gpu:
            Z, X = Variable(Z.float().cuda()), Variable(X.float().cuda())
        else:
            Z, X = Variable(Z.float()), Variable(X.float())

        ####process data

        Z = Z.contiguous()
        X = X.contiguous()

        ####contiguous to aviod gap in view
        # acts_diff = Z[:, :, :, :35]
        # actions = Z[:, :, :, 35:70]
        # acts_v = Z[:, :, :, 70:]
        #
        # cam_real = X[:, :, :6]
        # emoint = X[:, :, 6:7]
        # inicam = X[:, 0:1, :6]
        #
        # inicam = inicam.repeat((1, cam_real.shape[1], 1))

        # fake input test
        act_diff = torch.from_numpy(np.random.rand(10, 45, 6, 35)).float().cuda()
        action = torch.from_numpy(np.random.rand(10, 45, 6, 35)).float().cuda()
        act_v = torch.from_numpy(np.random.rand(10, 45, 6, 35)).float().cuda()
        cam_real = torch.from_numpy(np.random.rand(10, 45, 6)).float().cuda()
        inipos = torch.from_numpy(np.random.rand(10, 45, 2, 3)).float().cuda()
        aes_score = torch.from_numpy(np.random.rand(10, 45, 1)).float().cuda()
        emoint = torch.from_numpy(np.random.rand(10, 45, 1)).float().cuda()
        inicam = torch.from_numpy(np.random.rand(10, 45, 6)).float().cuda()


        """ Discriminator Pass"""
        # Predict

        cam_fake = G(act_diff, action, act_v, inipos, aes_score, emoint, inicam)
        D_real = D(cam_real)
        D_fake = D(cam_fake)

        # D loss
        loss_d = -torch.mean(log(D_real) + log(1 - D_fake))
        loss_d.backward()
        D_optimizer.step()
        reset_grad()

        """ Generator Pass"""

        # Sample from GS
        cam_fake = G(act_diff, action, act_v, inipos, aes_score, emoint, inicam)
        # Eval with D
        D_fake = D(cam_fake)
        loss_gan = -torch.mean(log(D_fake))

        # Char loss

        fake_offset = cam_fake[:, 1:, :] - cam_fake[:, :-1, :]
        real_offset = cam_real[:, 1:, :] - cam_real[:, :-1, :]

        loss_loc = CharbMKIJSEWT(cam_fake, cam_real)
        loss_direct = CharbDirect(fake_offset, real_offset)
        loss_tv = CharbTV(cam_fake)

        loss_char = loss_loc + loss_direct + loss_tv

        # Feature Loss
        loss_feat = vggfeature_loss(cam_fake, cam_real, vgg_models)

        # Combined
        loss_combined = args.lamb[0] * loss_char + args.lamb[1] * loss_feat + args.lamb[2] * loss_gan
        loss_combined.backward()
        G_optimizer.step()
        reset_grad()

        """ Inform """
        train_loss_d += loss_d
        step_loss_d += loss_d
        train_loss_g += loss_combined
        step_loss_g += loss_combined

        train_loss_char += loss_char * args.lamb[0]
        train_loss_feat += loss_feat * args.lamb[1]
        train_loss_gan += loss_gan * args.lamb[2]

        data_idx += len(Z)

        if (batch_idx + 1) % args.log_interval == 0:
            string_step = '{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss D step: {:.6f}\tLoss D train: {:.6f}\tLoss G step: {:.6f}\tLoss G train: {:.6f}\n'.format(
                pid, epoch, data_idx, len(data_loader.dataset),
                100. * (batch_idx + 1) / len(data_loader), step_loss_d / float(args.log_interval),
                train_loss_d / float(batch_idx + 1), step_loss_g / float(args.log_interval),
                train_loss_g / float(batch_idx + 1))

            for logger in args.loggers:
                logger(string_step)

            step_loss_g = 0.0
            step_loss_d = 0.0

        # Update lr D
        pre_lr_d = args.lr_d
        args.lr_d = args.lr_scheduler_d.update(args.lr_d)

        if pre_lr_d != args.lr_d:
            for param_group in D_optimizer.param_groups:
                param_group['lr'] = args.lr_d

        # Update lr G
        pre_lr_g = args.lr_g
        args.lr_g = args.lr_scheduler_g.update(args.lr_g)

        if pre_lr_g != args.lr_g:
            for param_group in G_optimizer.param_groups:
                param_group['lr'] = args.lr_g

    return [train_loss_g / float(len(data_loader)), train_loss_char / float(len(data_loader)),
            train_loss_feat / float(len(data_loader)), train_loss_gan / float(len(data_loader))]


def train_gan_loss(G, D, train_data, train_labels, args):
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size * args.sub_iters,
                                               shuffle=args.shuffle)

    if args.optimizer_type_d == 'SGD':
        D_optimizer = optim.SGD(D.parameters(), lr=args.lr_d,
                                momentum=args.optimizer_params_d['momentum'],
                                weight_decay=args.optimizer_params_d['weight_decay'])
    elif args.optimizer_type_d == 'ADAM':
        D_optimizer = optim.Adam(D.parameters(), lr=args.lr_d,
                                 weight_decay=args.optimizer_params_d['weight_decay'])

    if args.optimizer_type_g == 'SGD':
        G_optimizer = optim.SGD(G.parameters(), lr=args.lr_g,
                                momentum=args.optimizer_params_g['momentum'],
                                weight_decay=args.optimizer_params_g['weight_decay'])
    elif args.optimizer_type_g == 'ADAM':
        G_optimizer = optim.Adam(G.parameters(), lr=args.lr_g,
                                 weight_decay=args.optimizer_params_g['weight_decay'])

    # Create VGG model for loss function

    vgg_models = create_vgg_model(gpu=args.gpu)

    if args.gpu:
        D.cuda()
        G.cuda()

    for epoch in range(1, args.epochs + 1):
        epoch_time = time()
        train_loss = train_epoch(epoch, args, D, G, vgg_models, train_loader, D_optimizer, G_optimizer)
        epoch_time = time() - epoch_time

        string_epoch = "#############Epoch {}############" \
                       "Finished epoch in {:.2f} seconds.\n " \
                       "G Train loss: {:.6f}.\n" \
                       "G char loss: {:.6f}.\n" \
                       "G feat loss: {:.6f}.\n" \
                       "G gan loss: {:.6f}.\n"\
            .format(epoch, epoch_time, train_loss[0], train_loss[1], train_loss[2], train_loss[3])

        for logger in args.loggers:
            logger(string_epoch)

        if args.path_save is not None:
            D.save(join(args.path_save, 'w_d.' + str(epoch) + '.pt'))
            G.save(join(args.path_save, 'w_g.' + str(epoch) + '.pt'))

        # Update lr D
        pre_lr_d = args.lr_d
        args.lr_d = args.lr_scheduler_d.epoch_end(args.lr_d)

        if pre_lr_d != args.lr_d:
            for param_group in D_optimizer.param_groups:
                param_group['lr'] = args.lr_d

        # Update lr G
        pre_lr_g = args.lr_g
        args.lr_g = args.lr_scheduler_g.epoch_end(args.lr_g)

        if pre_lr_g != args.lr_g:
            for param_group in G_optimizer.param_groups:
                param_group['lr'] = args.lr_g
