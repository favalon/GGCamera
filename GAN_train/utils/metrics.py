from math import log10

import torch
from torch.nn import MSELoss


def PSNR(x, y, max_val=1.0):
    return -10 * log10(max_val / MSELoss(size_average=True)(x, y).data[0])


class Charbonnier():
    def __init__(self, epsilon=0.001, size_average=True):
        self.epsilon = epsilon ** 2
        self.size_average = size_average

    def __call__(self, x, y):
        l = torch.sqrt((x - y) ** 2 + self.epsilon)
        l = l.mean() if self.size_average else l.sum()

        return l
