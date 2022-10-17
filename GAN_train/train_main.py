from os import makedirs
from os.path import exists
import os
import numpy as np
import torch
from torch.nn import MSELoss

from models.GAN import Discriminator
from models.NetArchi_bc import Generator
from mufpt.loggers import console_logger, file_logger
from mufpt.lr_schedulers import lr_dummy_scheduler, lr_step_epoch_scheduler
from train_detail import train_gan_loss


class train_arguments_gan():
    DEFUALT_OPTIMIZER_PARAMS = {'SGD': {'momentum': 0.9, 'weight_decay': 0.0},
                                'ADAM': {'betas': (0.9, 0.999), 'weight_decay': 0.0}
                                }

    def __init__(self, lamb=None,
                 epochs=0, shuffle=True,
                 batch_size=32, sub_iters=1,
                 lr_d=0.01, lr_g=0.01,
                 optimizer_type_d='SGD', optimizer_type_g='SGD',
                 optimizer_params_d=None, optimizer_params_g=None,
                 lamb_gan_scheduler=None,
                 lamb_feat_scheduler=None,
                 lr_scheduler_d=None, lr_scheduler_g=None,
                 metrics={},
                 log_interval=1, gpu=False,
                 loggers=[], path_save=None, normdata=None):

        if isinstance(lamb, list):
            self.lamb = lamb

        if normdata is not None:
            actsnorm = normdata[:, :-12]
            actsnorm = np.split(actsnorm, 12, axis=1)
            actsnorm = np.concatenate((actsnorm[8], actsnorm[9]), axis=1)
            gtnorm = normdata[:, -2:]

            self.normact = torch.from_numpy(actsnorm)
            self.normgt = torch.from_numpy(gtnorm)

        self.epochs = epochs
        self.batch_size = batch_size
        self.sub_iters = sub_iters
        self.shuffle = shuffle
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.optimizer_type_d = optimizer_type_d

        if optimizer_params_d is None:
            self.optimizer_params_d = self.DEFUALT_OPTIMIZER_PARAMS[self.optimizer_type_d]
        else:
            self.optimizer_params_d = optimizer_params_d

        self.optimizer_type_g = optimizer_type_g

        if optimizer_params_g is None:
            self.optimizer_params_g = self.DEFUALT_OPTIMIZER_PARAMS[self.optimizer_type_g]
        else:
            self.optimizer_params_g = optimizer_params_g

        self.lamb_gan_scheduler = lr_dummy_scheduler()

        if lamb_gan_scheduler is not None:
            self.lamb_gan_scheduler = lamb_gan_scheduler

        self.lamb_feat_scheduler = lr_dummy_scheduler()

        if lamb_feat_scheduler is not None:
            self.lamb_feat_scheduler = lamb_feat_scheduler

        self.lr_scheduler_d = lr_dummy_scheduler()

        if lr_scheduler_d is not None:
            self.lr_scheduler_d = lr_scheduler_d

        self.lr_scheduler_g = lr_dummy_scheduler()

        if lr_scheduler_g is not None:
            self.lr_scheduler_g = lr_scheduler_g

        self.metrics = metrics
        self.log_interval = log_interval
        self.gpu = gpu
        self.loggers = [console_logger(), ] + loggers
        self.path_save = path_save


def load_all_data(path):

    with open(os.path.join(path, 'act_info_base_contain.npy'), 'rb') as f:
        act_info_base_contain = np.load(f)
    with open(os.path.join(path, 'act_info_diff_contain.npy'), 'rb') as f:
        act_info_diff_contain = np.load(f)
    with open(os.path.join(path, 'act_info_v_contain.npy'), 'rb') as f:
        act_info_v_contain = np.load(f)
    with open(os.path.join(path, 'init_camera_contain.npy'), 'rb') as f:
        init_camera_contain = np.load(f)
        init_camera_contain = \
            init_camera_contain.reshape((init_camera_contain.shape[0], init_camera_contain.shape[1], 6))
    with open(os.path.join(path, 'init_theta_contain.npy'), 'rb') as f:
        init_theta_contain = np.load(f)
    with open(os.path.join(path, 'emo_intensity_contain.npy'), 'rb') as f:
        emo_intensity_contain = np.load(f)
    with open(os.path.join(path, 'dis_diff_contain.npy'), 'rb') as f:
        dis_diff_contain = np.load(f)
    with open(os.path.join(path, 'position_contain.npy'), 'rb') as f:
        position_contain = np.load(f)
        position_contain = \
            position_contain.reshape((position_contain.shape[0], position_contain.shape[1], 6))
    with open(os.path.join(path, 'camera_data_contain.npy'), 'rb') as f:
        camera_data_contain = np.load(f)
        camera_data_contain = \
            camera_data_contain.reshape((camera_data_contain.shape[0], camera_data_contain.shape[1], 6))

    train_data = torch.cat((torch.from_numpy(act_info_base_contain), torch.from_numpy(act_info_diff_contain),
                            torch.from_numpy(act_info_v_contain)), dim=3)

    train_label = torch.cat((torch.from_numpy(init_camera_contain), torch.from_numpy(init_theta_contain),
                            torch.from_numpy(emo_intensity_contain), torch.from_numpy(dis_diff_contain),
                             torch.from_numpy(position_contain), torch.from_numpy(camera_data_contain)), dim=2)

    return train_data[:1780], train_label[:1780]


if __name__ == '__main__':

    # add training data

    # data loading path
    data_root = "GGCamera_data/processed_np"

    train_data, train_label = load_all_data(data_root)

    path_save = "GGCamera_data/checkpointGAN"

    batch_size = 10
    # train_acts_diff = torch.rand((1000, 45, 6, 35))
    # train_acts_v = torch.rand((1000, 45, 6, 35))
    # train_actions = torch.rand((1000, 45, 6, 35))
    #
    # train_data = torch.cat((train_acts_diff, train_actions, train_acts_v), dim=3)
    #
    # train_label = torch.rand((1000, 45, 8))

    in_shape = 35
    out_shape = 6  # 6 + 1 (camera pos + aes score)

    print("Training Network Input Shape : {}, Output Shape : {}".format(in_shape, out_shape))

    train_data = train_data.float()
    train_label = train_label.float()

    # Create output dir
    if not exists(path_save):
        makedirs(path_save)

    G = Generator(batch_size, in_shape, out_shape)
    D = Discriminator()

    G.load(os.path.join(path_save, "w_g.1000.pt"))
    D.load(os.path.join(path_save, "w_d.1000.pt"))

    lamb_char = 10
    lamb_gan = 0.3
    lamb_feat = 0.25

    lambs = [lamb_char, lamb_feat, lamb_gan]

    # Train model
    lr_d = 10 ** (-4)
    lr_g = 10 ** (-4)

    # Train model
    train_args = train_arguments_gan(lamb=lambs,
                                     epochs=1000, batch_size=batch_size,
                                     sub_iters=1, shuffle=False,
                                     lr_d=lr_d, lr_g=lr_g,
                                     lr_scheduler_d=lr_step_epoch_scheduler(steps=[30, 40], lrs=[lr_g / 2, lr_g / 10]),
                                     lr_scheduler_g=lr_step_epoch_scheduler(steps=[30, 40], lrs=[lr_d / 2, lr_d / 10]),
                                     optimizer_type_d='ADAM', optimizer_params_d={'weight_decay': 0.001},
                                     optimizer_type_g='ADAM', optimizer_params_g={'weight_decay': 0.0001},
                                     log_interval=train_data.size()[0] / (batch_size * 100), gpu=True,
                                     metrics={'mse': MSELoss(size_average=True)},
                                     loggers=[file_logger(path_save + '/log.txt', False)], path_save=path_save)
    train_gan_loss(G, D, train_data, train_label, train_args)
