import copy

import torch
import torch.nn as nn
from torchvision import models

from GAN_train.utils.metrics import Charbonnier


def cal_aes_loss(x, y):
    x = x.reshape((x.shape[0], x.shape[1], 1))
    weight = torch.ones(x.shape[0], x.shape[1], x.shape[2]).type_as(x)
    weight[:, :10, :] = 2
    weight[:, -10:, :] = 2
    eps = 1e-6
    loss = torch.sqrt((x - y) ** 2 + eps)
    loss = loss * weight
    loss = loss.mean()
    return loss


def cal_direct_loss(x, y):
    eps = 1e-6
    signx = torch.sign(x)
    signy = torch.sign(y)
    weight = torch.abs(signy - signx)
    weight = weight + 1
    loss = torch.sqrt((x - y) ** 2 + eps)
    loss = loss * weight
    loss = loss.mean()
    return loss


def cal_general_mse_loss(x, y):
    weight = torch.ones(x.shape[0], x.shape[1], x.shape[2]).type_as(x)
    weight[:, :10, :] = 2
    weight[:, -10:, :] = 2
    eps = 1e-6
    loss = torch.sqrt((x - y) ** 2 + eps)
    loss = loss * weight
    loss = loss.mean()
    return loss


def MSELossWT(x, y):
    weight = torch.ones(x.shape[0], x.shape[1], x.shape[2]).type_as(x)
    weight[:, 0, :] = 5
    loss = (x - y) ** 2
    loss = loss * weight
    loss = loss.sum()
    return loss


def L1LossWT(x, y):
    weight = torch.ones(x.shape[0], x.shape[1], x.shape[2]).type_as(x)
    weight[:, 0, :] = 5
    loss = torch.abs(x - y)
    loss = loss * weight
    loss = loss.sum()
    return loss


def DirectSign(x, y):
    ###based on offset

    signx = torch.sign(x)
    signy = torch.sign(y)
    weight = torch.abs(signy - signx)
    weight = weight + 1
    loss = torch.abs(x - y)
    loss = loss * weight
    loss = loss.sum()
    return loss


def TVloss(x):
    #####total variation
    ###weakly-supervised

    x_front = x[:, 0:-2, :]
    x_center = x[:, 1:-1, :]
    x_after = x[:, 2:, :]

    loss = torch.abs(x_center - x_front) + torch.abs(x_center - x_after)
    loss = loss.sum()

    return loss


def CharbMSEWT(x, y):
    weight = torch.ones(x.shape[0], x.shape[1], x.shape[2]).type_as(x)
    weight[:, 0, :] = 5
    eps = 1e-6
    loss = torch.sqrt((x - y) ** 2 + eps)
    loss = loss * weight
    loss = loss.mean()
    return loss


def CharbDirect(x, y):
    #####based on offset
    eps = 1e-6
    signx = torch.sign(x)
    signy = torch.sign(y)
    weight = torch.abs(signy - signx)
    weight = weight + 1
    loss = torch.sqrt((x - y) ** 2 + eps)
    loss = loss * weight
    loss = loss.mean()
    return loss


def CharbTV(x):
    eps = 1e-6
    x_front = x[:, 0:-2, :]
    x_center = x[:, 1:-1, :]
    x_after = x[:, 2:, :]
    loss = torch.abs(x_center - x_front) + torch.abs(x_center - x_after) + eps
    loss = loss.mean()
    return loss


def create_vgg_model(end_layers=[8, 15], path=None, gpu=False):
    vgg16 = models.vgg16(pretrained=False)

    vgg16.load_state_dict(torch.load("pre_train_models/vgg16-397923af.pth"))

    vgg16 = vgg16.features

    vgg_models = []

    for lay in end_layers:
        if (gpu):
            vgg = cut_layers(vgg16, lay)
            vgg.cuda()
        else:
            vgg = cut_layers(vgg16, lay)
        # Make sure you dont differentiate VGG parameters during training
        for param in vgg.parameters():
            param.requires_grad = False
        vgg_models.append(vgg)

    return vgg_models


def cut_layers(vgg, end_layer, use_maxpool=True):
    """
        [1] uses the output of vgg16 relu2_2 layer as a loss function (layer8 on PyTorch default vgg16 model).
        This function expects a vgg16 model from PyTorch and will return a custom version up until layer = end_layer
        that will be used as our loss function.
    """

    vgg = copy.deepcopy(vgg)
    model = nn.Sequential()

    i = 0
    for layer in list(vgg):

        if i > end_layer:
            break

        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            if use_maxpool:
                model.add_module(name, layer)
            else:
                avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                model.add_module(name, avgpool)
        i += 1
    return model


def vggfeature_loss(fake, gt, vgg_models):
    charb_loss = Charbonnier(epsilon=0.001, size_average=True)

    ###rescale to 0-1 for each image
    ####amin-amax in newest ver--sad :(
    fakemin, _ = torch.min(fake, dim=1)
    fakemin = fakemin.reshape(fake.shape[0], 1, fake.shape[2])
    fakemin = fakemin.repeat(1, fake.shape[1], 1)

    fakemax, _ = torch.max(fake, dim=1)
    fakemax = fakemax.reshape(fake.shape[0], 1, fake.shape[2])
    fakemax = fakemax.repeat(1, fake.shape[1], 1)

    fakerange = fakemax - fakemin
    fakerange[fakemax == fakemin] = -1
    fakerange = 1 / fakerange
    fakerange[fakerange < 0] = 0

    fake = (fake - fakemin) * fakerange

    ##########processing nan if no move
    ####adding elipson
    ############nan to num in 1.8 :(

    # fake = (fake - fakemin) / (fakemax-fakemin)
    # fake=torch.nan_to_num(fake,nan=0)
    fake = fake + 0.00001

    gtmin, _ = torch.min(gt, dim=1)
    gtmin = gtmin.reshape(gt.shape[0], 1, gt.shape[2])
    gtmin = gtmin.repeat(1, gt.shape[1], 1)

    gtmax, _ = torch.max(gt, dim=1)
    gtmax = gtmax.reshape(gt.shape[0], 1, gt.shape[2])
    gtmax = gtmax.repeat(1, gt.shape[1], 1)

    gtrange = gtmax - gtmin
    gtrange[gtmax == gtmin] = -1
    gtrange = 1 / gtrange
    gtrange[gtrange < 0] = 0

    gt = (gt - gtmin) * gtrange

    ##########processing nan if no move
    ####adding elipson
    ############nan to num in 1.8 :(

    # gt=(gt - gtmin) / (gtmax-gtmin)
    # gt=torch.nan_to_num(gt,nan=0)
    gt = gt + 0.00001

    # Need to replicate the gray channel across all expected colors channels
    fake = (fake - 0.449) / 0.225
    gt = (gt - 0.449) / 0.225
    fake = fake.unsqueeze(1).repeat(1, 3, 1, 1)
    gt = gt.unsqueeze(1).repeat(1, 3, 1, 1)

    vgg_loss = 0
    for vgg in vgg_models:
        vgg_loss_inp = vgg(fake)
        vgg_loss_tgt = vgg(gt)
        vgg_loss += charb_loss(vgg_loss_inp, vgg_loss_tgt)

    return vgg_loss


def vggfeature_loss_v18(fake, gt, vgg_models):
    charb_loss = Charbonnier(epsilon=0.001, size_average=True)

    ###rescale to 0-1 for each image
    ####amin-amax in newest ver--sad :(
    fakemin, _ = torch.min(fake, dim=1)
    fakemin = fakemin.reshape(fake.shape[0], 1, fake.shape[2])
    fakemin = fakemin.repeat(1, fake.shape[1], 1)

    fakemax, _ = torch.max(fake, dim=1)
    fakemax = fakemax.reshape(fake.shape[0], 1, fake.shape[2])
    fakemax = fakemax.repeat(1, fake.shape[1], 1)

    ##########processing nan if no move
    ####adding elipson
    ############nan to num in 1.8 :(

    fake = (fake - fakemin) / (fakemax - fakemin)
    fake = torch.nan_to_num(fake, nan=0)
    fake = fake + 0.00001

    gtmin, _ = torch.min(gt, dim=1)
    gtmin = gtmin.reshape(gt.shape[0], 1, gt.shape[2])
    gtmin = gtmin.repeat(1, gt.shape[1], 1)

    gtmax, _ = torch.max(gt, dim=1)
    gtmax = gtmax.reshape(gt.shape[0], 1, gt.shape[2])
    gtmax = gtmax.repeat(1, gt.shape[1], 1)

    ##########processing nan if no move
    ####adding elipson
    ############nan to num in 1.8 :(

    gt = (gt - gtmin) / (gtmax - gtmin)
    gt = torch.nan_to_num(gt, nan=0)
    gt = gt + 0.00001

    # Need to replicate the gray channel across all expected colors channels
    fake = (fake - 0.449) / 0.225
    gt = (gt - 0.449) / 0.225
    fake = fake.unsqueeze(1).repeat(1, 3, 1, 1)
    gt = gt.unsqueeze(1).repeat(1, 3, 1, 1)

    vgg_loss = 0
    for vgg in vgg_models:
        vgg_loss_inp = vgg(fake)
        vgg_loss_tgt = vgg(gt)
        vgg_loss += charb_loss(vgg_loss_inp, vgg_loss_tgt)

    return vgg_loss


def CharbMSEWT4RoT(x, y):
    weight_ini = torch.ones(x.shape[0], x.shape[1], x.shape[2]).type_as(x)
    weight_ini[:, 0, :] = 5

    weight_cam = torch.ones(x.shape[0], x.shape[1], x.shape[2]).type_as(x)
    weight_cam = weight_cam * 5
    weight_cam[:, :, 0] = 1
    weight_cam[:, :, 1] = 1
    eps = 1e-6
    loss = (x - y) ** 2
    loss = torch.sqrt(loss * weight_cam + eps)
    loss = loss * weight_ini
    loss = loss.mean()
    return loss


def CharbMSEWT4RoT_single(x, y):
    ####fake,real

    weight_cam = torch.ones(x.shape[0], x.shape[1]).type_as(x)
    weight_cam = weight_cam * 5
    weight_cam[:, 0] = 1
    weight_cam[:, 1] = 1
    eps = 1e-6
    loss = (x - y) ** 2 + eps
    weight_v = torch.ones(x.shape[0], x.shape[1]).type_as(x)
    weight_v[:, 1] = ((x[:, 1] < -2.6527) + (x[:, 1] > 1.92)).float()
    weight_v[:, 1] = weight_v[:, 1] * 4 + 1

    weight_v[:, 0] = ((x[:, 0] < -2.3) + (x[:, 0] > 2.3)).float()
    weight_v[:, 0] = weight_v[:, 0] * 4 + 1

    loss = torch.sqrt(loss) * weight_cam * weight_v
    loss = loss.mean()
    return loss


def BodyRegionDis(x, y):
    #####log value of isOnCanvas

    eps = 1e-6
    loss = (x - y) ** 2
    loss = loss + eps
    loss = torch.sum(loss, dim=(1, 2))
    loss = loss.mean()
    return loss


def BodyRegionDisContras(x, y):
    #####log value of isOnCanvas

    eps = 1e-6
    loss = x * (1 - y) + (1 - x) * y
    loss = loss + eps
    loss = torch.sum(loss, dim=(1, 2))
    loss = loss.mean()
    return loss


def BodyShapeDis(x, y):
    ####penalize out-canvas points
    penalize_x = torch.ones(x.shape).type_as(x)
    penalize_x[:, :, :, 0] = -960
    penalize_x[:, :, :, 1] = -540

    log_x = torch.sum(x, dim=3) <= 0
    log_x = log_x.type_as(x)
    log_x = log_x.unsqueeze(3)
    penalize_x = penalize_x * log_x

    x = x + penalize_x

    penalize_y = torch.ones(y.shape).type_as(y)
    penalize_y[:, :, :, 0] = -960
    penalize_y[:, :, :, 1] = -540

    log_y = torch.sum(y, dim=3) <= 0
    log_y = log_y.type_as(y)
    log_y = log_y.unsqueeze(3)
    penalize_y = penalize_y * log_y

    y = y + penalize_y

    ##########oncanvs points
    eps = 1e-6

    headset_x_h = x[:, 0, 0:1, :] - x[:, 0, 1:2, :]
    armset_x_h = x[:, 2, :-1, :] - x[:, 3, :-1, :]
    legset_x_h = x[:, 4, 1:, :] - x[:, 5, 1:, :]

    headset_x_v = x[:, 0, 2:3, :] - x[:, 0, 3:4, :]
    torsor_x_v = x[:, 1, 2:-1, :] - x[:, 1, 3:, :]
    armset1_x_v = x[:, 2, 0:-2, :] - x[:, 2, 1:-1, :]
    armset2_x_v = x[:, 3, 0:-2, :] - x[:, 3, 1:-1, :]
    legset1_x_v = x[:, 4, 1:-1, :] - x[:, 4, 2:, :]
    legset2_x_v = x[:, 5, 1:-1, :] - x[:, 5, 2:, :]

    shape_x = torch.cat((headset_x_h, armset_x_h, legset_x_h, headset_x_v, torsor_x_v,
                         armset1_x_v, armset2_x_v, legset1_x_v, legset2_x_v), dim=1)

    shape_x = torch.sum(shape_x ** 2, dim=2) + eps
    shape_x = torch.sqrt(shape_x)

    headset_y_h = y[:, 0, 0:1, :] - y[:, 0, 1:2, :]
    armset_y_h = y[:, 2, :-1, :] - y[:, 3, :-1, :]
    legset_y_h = y[:, 4, 1:, :] - y[:, 5, 1:, :]

    headset_y_v = y[:, 0, 2:3, :] - y[:, 0, 3:4, :]
    torsor_y_v = y[:, 1, 2:-1, :] - y[:, 1, 3:, :]
    armset1_y_v = y[:, 2, 0:-2, :] - y[:, 2, 1:-1, :]
    armset2_y_v = y[:, 3, 0:-2, :] - y[:, 3, 1:-1, :]
    legset1_y_v = y[:, 4, 1:-1, :] - y[:, 4, 2:, :]
    legset2_y_v = y[:, 5, 1:-1, :] - y[:, 5, 2:, :]

    shape_y = torch.cat((headset_y_h, armset_y_h, legset_y_h, headset_y_v, torsor_y_v,
                         armset1_y_v, armset2_y_v, legset1_y_v, legset2_y_v), dim=1)

    shape_y = torch.sum(shape_y ** 2, dim=2) + eps
    shape_y = torch.sqrt(shape_y)

    loss = torch.sqrt((shape_x - shape_y) ** 2 + eps)
    loss = torch.sum(loss, dim=1) / 1101

    loss = loss.mean()

    return loss


def BodyRegionLocDis(x, y):
    '''
    x=x.data.cpu().numpy()
    for pose in x:
        pose = pose.reshape(-1,2)
        plt.scatter(pose[:, 0], pose[:, 1])
        plt.show()
    '''

    ####penalize out-canvas points
    penalize_x = torch.ones(x.shape).type_as(x)
    penalize_x[:, :, :, 0] = -960
    penalize_x[:, :, :, 1] = -540

    log_x = torch.sum(x, dim=3) <= 0
    log_x = log_x.type_as(x)
    log_x = log_x.unsqueeze(3)
    penalize_x = penalize_x * log_x

    x = x + penalize_x

    penalize_y = torch.ones(y.shape).type_as(y)
    penalize_y[:, :, :, 0] = -960
    penalize_y[:, :, :, 1] = -540

    log_y = torch.sum(y, dim=3) <= 0
    log_y = log_y.type_as(y)
    log_y = log_y.unsqueeze(3)
    penalize_y = penalize_y * log_y

    y = y + penalize_y

    eps = 1e-6
    loss = torch.sum((x - y) ** 2 + eps, dim=3)
    loss = torch.sqrt(loss) / 1101
    loss = torch.sum(loss, dim=(1, 2))
    return loss.mean()


def CharbMSEWT4RoT_single_offset(x):
    ######make sure use norm data, so y thres based on norm value
    ####fake-real

    weight_cam = torch.ones(x.shape[0], x.shape[1]).type_as(x)
    weight_cam = weight_cam * 5
    weight_cam[:, 0] = 1
    weight_cam[:, 1] = 1
    eps = 1e-6

    loss = torch.abs(x) + eps

    loss = loss * weight_cam
    loss = loss.mean()
    return loss


if __name__ == '__main__':
    a = torch.randn((10, 45, 6)) * 0.01

    gt = torch.randn((10, 45, 5)) * 0.01
    add2 = torch.ones((10, 45, 1)) * 2
    gt = torch.cat((gt, add2), dim=2)

    vggs = create_vgg_model()
    dis1 = vggfeature_loss(a, gt, vggs)
    dis2 = vggfeature_loss_v18(a, gt, vggs)
    print(dis1)
    print(dis2)
