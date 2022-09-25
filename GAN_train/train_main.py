from os import makedirs
from os.path import exists

import torch

from models.GAN import Discriminator
from models.NetArchi_bc import Generator

if __name__ == '__main__':

    # add training data

    path_save = "../GANtrain/checkpointGAN"

    batch_size = 10
    train_acts_diff = torch.rand((1000, 45, 6, 35))
    train_acts_v = torch.rand((1000, 45, 6, 35))
    train_actions = torch.rand((1000, 45, 6, 35))

    train_data = torch.cat((train_acts_diff, train_actions, train_acts_v), dim=3)

    train_label = torch.rand((1000, 45, 8))

    in_shape = train_actions.shape[-1]
    out_shape = train_label.shape[-1] - 2  # 这里其实是6

    print("Training Network Input Shape : {}, Output Shape : {}".format(in_shape, out_shape))

    train_data = train_data.float()
    train_label = train_label.float()

    # Create output dir
    if not exists(path_save):
        makedirs(path_save)

    G = Generator(batch_size, in_shape, out_shape)
    D = Discriminator()
