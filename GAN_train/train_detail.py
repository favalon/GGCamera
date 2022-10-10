from os import getpid
from os.path import join
from time import time

import numpy as np
import torch.optim as optim
from torch.autograd import Variable

from general_loss import *


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

    train_loss_general = 0.0
    train_loss_feat = 0.0
    train_loss_gan = 0.0

    data_idx = 0

    for batch_idx, (Z, X) in enumerate(data_loader):

        # Sample Z, X
        if args.gpu:
            Z, X = Variable(Z.float().cuda()), Variable(X.float().cuda())
        else:
            Z, X = Variable(Z.float()), Variable(X.float())

        # process data
        Z = Z.contiguous()
        X = X.contiguous()

        if batch_idx == 42:
            print("")

        # contiguous to aviod gap in view
        # init_camera_is = [0:6]
        # init_theta_s = [6]
        # emo_intensity_is = [7]
        # dis_diff_is = [8]
        # position_is = [9:15]
        # camera_data_is = [15:21]

        # fake input test
        act_diff = Z[:, :, :, :35]
        action = Z[:, :, :, 35:70]
        act_v = Z[:, :, :, 70:]
        inicam = X[:, :, 0:6]
        initheta = X[:, :, 6:7]
        emoint = X[:, :, 7:8]
        inipos = X[:, :, 9:15]
        cam_real = X[:, :, 15:21]

        # act_v = torch.from_numpy(np.zeros(act_v.shape)).float().cuda()
        # initheta = torch.from_numpy(np.zeros(initheta.shape)).float().cuda()
        # inipos = torch.from_numpy(np.zeros(inipos.shape)).float().cuda()

        aes_score_real = torch.from_numpy(np.zeros((cam_real.shape[0], 100, 1))).float().cuda()
        cam_real = torch.cat((cam_real, aes_score_real), 2)
        # cam_real = torch.from_numpy(np.random.rand(10, 45, 7)).float().cuda()

        """ Discriminator Pass"""
        # Predict

        cam_fake = G(act_diff, action, act_v, inipos, initheta, emoint, inicam)
        D_real = D(cam_real)
        D_fake = D(cam_fake)

        # D loss
        loss_d = -torch.mean(log(D_real) + log(1 - D_fake))
        loss_d.backward()
        # torch.nn.utils.clip_grad_norm_(D.parameters(), -1, 1)
        D_optimizer.step()
        reset_grad()

        """ Generator Pass"""

        # Sample from GS
        cam_fake = G(act_diff, action, act_v, inipos, initheta, emoint, inicam)
        # Eval with D
        D_fake = D(cam_fake)
        loss_gan = -torch.mean(log(D_fake))

        # general loss

        fake_offset = cam_fake[:, 1:, :6] - cam_fake[:, :-1, :6]
        real_offset = cam_real[:, 1:, :6] - cam_real[:, :-1, :6]

        # loss_aes = cal_aes_loss(x=cam_fake[:, :, 6], y=aes_score_real)
        loss_direct = cal_direct_loss(x=fake_offset, y=real_offset)
        loss_mse = cal_general_mse_loss(x=cam_fake[:, :, :6], y=cam_real[:, :, :6])

        loss_general = loss_direct + loss_mse # + loss_aes

        # Feature Loss
        loss_feat = vggfeature_loss(cam_fake, cam_real, vgg_models)

        # Combined
        loss_combined = args.lamb[0] * loss_general + \
                        args.lamb[1] * loss_feat + \
                        args.lamb[2] * loss_gan

        loss_combined.backward()
        # torch.nn.utils.clip_grad_norm_(G.parameters(), -1, 1)
        G_optimizer.step()
        reset_grad()

        """ Inform """
        train_loss_d += loss_d
        step_loss_d += loss_d
        train_loss_g += loss_combined
        step_loss_g += loss_combined

        train_loss_general += loss_general * args.lamb[0]
        train_loss_feat += loss_feat * args.lamb[1]
        train_loss_gan += loss_gan * args.lamb[2]

        data_idx += len(Z)

        if (batch_idx + 1) % args.log_interval == 0:
            string_step = '{}\t' \
                          'Train Epoch: {} [{}/{} ({:.0f}%)]\t' \
                          'Loss D step: {:.6f}\t' \
                          'Loss D train: {:.6f}\t' \
                          'Loss G step: {:.6f}\t' \
                          'Loss G train: {:.6f}\n' \
                .format(pid, epoch, data_idx, len(data_loader.dataset),
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

    return [train_loss_g / float(len(data_loader)), train_loss_general / float(len(data_loader)),
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
                       "G general loss: {:.6f}.\n" \
                       "G feat loss: {:.6f}.\n" \
                       "G gan loss: {:.6f}.\n" \
            .format(epoch, epoch_time, train_loss[0], train_loss[1], train_loss[2], train_loss[3])

        for logger in args.loggers:
            logger(string_epoch)

        if args.path_save is not None and epoch % 50 ==0:
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
