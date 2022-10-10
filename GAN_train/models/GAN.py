from __future__ import division

import torch
import torch.nn as nn

from mufpt.base_model import BaseModel

T = torch
if torch.cuda.is_available():
    T = torch.cuda


class Discriminator(BaseModel):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.feat_branch = nn.Sequential(
            nn.Linear(700, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.5, inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.5, inplace=True),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.5, inplace=True),
            nn.Linear(64, 1),
        )

        self.sigmo = nn.Sigmoid()

    def forward(self, cammove):
        cammove = cammove.reshape((cammove.shape[0], cammove.shape[1] * cammove.shape[2]))

        camfeat = self.feat_branch(cammove)
        out = self.sigmo(camfeat)

        return out.view(-1, 1)


class SiameseDisCat(BaseModel):
    def __init__(self):
        super(SiameseDisCat, self).__init__()

        self.shared_branch = nn.Sequential(
            nn.Linear(270, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 30),
        )

        self.joinmch = nn.Linear(60, 1)
        ####sigmo but not softmax---sigmo independent prob but softmax sum to 1
        self.sigmo = nn.Sigmoid()

    def forward(self, camfake, camreal):
        camfake = camfake.reshape((camfake.shape[0], camfake.shape[1] * camfake.shape[2]))
        camreal = camreal.reshape((camreal.shape[0], camreal.shape[1] * camreal.shape[2]))

        fake_out = self.shared_branch(camfake)
        real_out = self.shared_branch(camreal)

        out = torch.cat((fake_out, real_out), dim=1)
        out = self.joinmch(out)
        out = self.sigmo(out)

        return out.view(-1, 1)


class SiameseDisEdu(BaseModel):
    def __init__(self):
        super(SiameseDisEdu, self).__init__()

        self.shared_branch = nn.Sequential(
            nn.Linear(270, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 30),
        )

        self.distance = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, camfake, camreal):
        camfake = camfake.reshape((camfake.shape[0], camfake.shape[1] * camfake.shape[2]))
        camreal = camreal.reshape((camreal.shape[0], camreal.shape[1] * camreal.shape[2]))

        fake_out = self.shared_branch(camfake)
        real_out = self.shared_branch(camreal)

        out = self.distance(fake_out, real_out)
        out = 1 - out

        return out.view(-1, 1)


if __name__ == '__main__':
    insalfeat = torch.rand((10, 45, 6))
    insalfeat2 = torch.rand((10, 45, 6))
    dis = Discriminator()
    print(dis(insalfeat))
    dis2 = SiameseDisCat()
    print(dis2(insalfeat, insalfeat))
    dis3 = SiameseDisEdu()
    print(dis3(insalfeat, insalfeat))
