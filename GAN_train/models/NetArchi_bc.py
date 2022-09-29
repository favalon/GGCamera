from __future__ import division

import torch
import torch.nn as nn

from mufpt.base_model import BaseModel

T = torch
if torch.cuda.is_available():
    T = torch.cuda


class basic_linear(nn.Module):

    def __init__(self, insize, outsize):
        super(basic_linear, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(insize, outsize),
            nn.LayerNorm(outsize),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(0.2, inplace=True),

        )

    def forward(self, input):
        output = self.layer(input)
        return output


class bodyreg_branch(nn.Module):

    def __init__(self, infeat):
        super(bodyreg_branch, self).__init__()
        self.linear1 = basic_linear(infeat, 64)
        self.linear2 = basic_linear(64, 256)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        self.linear3 = basic_linear(128, 128)
        self.linear4 = basic_linear(128, 128)
        self.meanpool = nn.AvgPool1d(2, stride=2)

    def forward(self, input):
        output = self.linear1(input)
        output = self.linear2(output)
        output = self.maxpool(output)
        output = self.linear3(output)
        output = self.linear4(output)
        output = self.meanpool(output)

        return output


class SalEncoder(BaseModel):
    def __init__(self, infeat):
        super(SalEncoder, self).__init__()

        self.headbranch = bodyreg_branch(infeat)
        self.torsorbranch = bodyreg_branch(infeat)
        self.armbranch = bodyreg_branch(infeat)
        self.legbranch = bodyreg_branch(infeat)
        self.fullbranch = basic_linear(384, 512)
        self.fc = nn.Linear(512, 384)

        self.softmax = nn.Softmax()

    def forward(self, act_diff):
        ####action (B,N,6,infeat=35)

        head, torsor, larm, rarm, lleg, rleg = torch.chunk(act_diff, 6, dim=2)

        head = self.headbranch(head.squeeze(2))
        torsor = self.torsorbranch(torsor.squeeze(2))
        larm = self.armbranch(larm.squeeze(2))
        rarm = self.armbranch(rarm.squeeze(2))
        lleg = self.legbranch(lleg.squeeze(2))
        rleg = self.legbranch(rleg.squeeze(2))

        ####concat
        fullbody = torch.cat((head, torsor, larm, rarm, lleg, rleg), dim=2)
        fullbody = self.fullbranch(fullbody)
        fullmask = self.fc(fullbody)
        fullmask = self.softmax(fullmask)
        head, torsor, larm, rarm, lleg, rleg = torch.chunk(fullmask, 6, dim=2)

        fullmask = torch.cat((head.unsqueeze(2), torsor.unsqueeze(2), larm.unsqueeze(2), rarm.unsqueeze(2),
                              lleg.unsqueeze(2), rleg.unsqueeze(2)), dim=2)

        return fullmask.contiguous(), fullbody.contiguous()


class TrajDecoder(BaseModel):

    def __init__(self, batch, inactfeat, incamfeat, aes_shape=1):
        super(TrajDecoder, self).__init__()

        self.actionbranch1 = basic_linear(inactfeat, 64)
        self.actmove = basic_linear(inactfeat, 64)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        self.actionbranch2 = basic_linear(256, 256)
        self.actionbranch3 = basic_linear(256, 512)

        self.inicambranch = basic_linear(incamfeat, 64)
        # self.inicambranch2 = basic_linear(128, 128)

        self.emobranch = basic_linear(128, 128)
        self.thetabranch = basic_linear(128, 128)
        self.iniposbranch = basic_linear(6, 256)

        self.sum_pa_branch = basic_linear(256, 512)

        self.rnn1 = nn.GRU(bidirectional=False, hidden_size=512, input_size=512, num_layers=1, batch_first=True)
        self.h0 = nn.Parameter(torch.randn(1, batch, 512).type(T.FloatTensor), requires_grad=True)

        self.dropout = nn.Dropout(p=0.2)
        self.sum_g_c_branch = basic_linear(512, 192)

        self.outbranch = basic_linear(512, 128)

        self.meanpool = nn.AvgPool1d(2, stride=2)

        self.outbranch2 = basic_linear(128, 64)

        self.fc = nn.Linear(64, incamfeat + 1)

    def forward(self, action, act_v, salmask, salfeat, inipos, initheta, emoint, inicam):
        # action(B,N,6,30), salfeat(B,N,6,64),
        # inipos(B, N, 2, 3) initheta(B, N, 1) emoint(B,N,1),inicam(B,N,6)

        action = self.actionbranch1(action)
        act_v = self.actmove(act_v)
        action = action + act_v  # [B, N, 6, 64]

        action = action * salmask
        action = action.reshape((action.shape[0], action.shape[1], action.shape[2] * action.shape[3]))  # [B, N, 384]
        action = self.maxpool(action)  # [B, N, 192]

        inicam = self.inicambranch(inicam)  # [B, N, 64]

        initheta = torch.cat(  # [B, N, 128]
            (initheta.repeat((1, 1, 64)), torch.randn(action.shape[0], action.shape[1], 64).type_as(initheta)), dim=2)
        initheta = self.thetabranch(initheta)  # [B, N, 128]
        initheta = self.maxpool(initheta)  # [B, N, 64]

        ini_set = torch.cat((inicam, initheta), dim=2)  # [B, N, 64]
        ini_set = self.maxpool(ini_set)
        action = torch.cat((action, ini_set), dim=2)  # [B, N, 256]

        inipos = inipos.reshape((inipos.shape[0], inipos.shape[1], inipos.shape[2] * inipos.shape[3]))
        inipos = self.iniposbranch(inipos)  # [B, N, 256]

        # P + A
        action = action + inipos  # [B, N, 256]
        action = self.sum_pa_branch(action)  # [B, N, 512]
        action = action + salfeat  # [B, N, 512]

        # E1
        action = action * emoint  # [B, N, 512]
        out, h1 = self.rnn1(action, self.h0)  # [B, N, 512]
        out = self.dropout(out)  # [B, N, 512]
        out = self.sum_g_c_branch(out)  # [B, N, 192]
        out = torch.cat((out, inicam), dim=2)  # [B, N, 256]
        out = torch.cat((out, inipos), dim=2)  # [B, N, 512]
        out = out * emoint  # [B, N, 512]
        out = self.outbranch(out)  # [B, N, 128]
        out = self.meanpool(out)  # [B, N, 64]
        out = torch.cat((out, inicam), dim=2)  # [B, N, 128]
        out = self.outbranch2(out)  # [B, N, 64]
        out = self.fc(out)  # [B, N, 6]

        return out.contiguous()


class Generator(BaseModel):
    def __init__(self, batch, in_shape, out_shape):
        super(Generator, self).__init__()

        self.sal_encoder = SalEncoder(in_shape)
        self.tj_decoder = TrajDecoder(batch, in_shape, out_shape)

    def forward(self, act_diff, action, act_v, inipos, initheta, emoint, inicam):
        outmask, outfeat = self.sal_encoder(act_diff)
        output = self.tj_decoder(action, act_v, outmask, outfeat, inipos, initheta, emoint, inicam)
        return output


class basic_linear_rot(nn.Module):

    def __init__(self, insize, outsize):
        super(basic_linear_rot, self).__init__()
        self.linear = nn.Linear(insize, outsize)
        self.atten = nn.Softmax()
        self.norm = nn.LayerNorm(outsize)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, input):
        output = self.linear(input)
        weight = self.atten(output)
        output = output + output * weight
        output = self.norm(output.unsqueeze(dim=1))
        output = output.squeeze(dim=1)

        output = self.relu(output)
        return output


'''
class basic_linear_rot2(nn.Module):

    def __init__(self, insize,outsize):
        super(basic_linear_rot2, self).__init__()
        self.linear = nn.Linear(insize, outsize)
        self.atten=nn.Softmax()
        #self.norm=nn.LayerNorm(outsize)
        self.relu=nn.LeakyReLU(0.2, inplace=True)


    def forward(self, input):
        output = self.linear(input)
        weight=self.atten(output)
        output=output*weight
        output=self.relu(output)

        return output

class basic_linear_rot_residual(nn.Module):

    def __init__(self, insize,outsize):
        super(basic_linear_rot_residual, self).__init__()
        self.linear = nn.Linear(insize, outsize)
        self.atten=nn.Softmax()
        self.relu=nn.LeakyReLU(0.2, inplace=True)


    def forward(self, input):
        output = self.linear(input)
        weight=self.atten(output)
        output=output*weight
        output=self.relu(output)
        return output

'''


class NetRoT(BaseModel):
    def __init__(self, in_shape_pose, cam_shape):
        super(NetRoT, self).__init__()

        self.feat1 = basic_linear_rot(in_shape_pose, 128)
        self.feat2 = basic_linear_rot(cam_shape, 64)
        self.feat3 = basic_linear_rot(128, 256)

        self.hidden1 = basic_linear_rot(256, 512)

        self.hidden2 = basic_linear_rot(512, 1024)

        self.hidden3 = basic_linear_rot(1024, 512)

        self.hidden4 = basic_linear_rot(512, 256)

        self.hidden5 = basic_linear_rot(256, 128)

        self.hidden6 = basic_linear_rot(128, 64)

        self.hidden7 = basic_linear_rot(64, 30)

        self.out = nn.Linear(30, cam_shape)

    def forward(self, actpose, camseq):
        actpose = self.feat1(actpose)
        camseq = self.feat2(camseq)
        camseq = camseq.repeat(1, 2)
        out = actpose + camseq
        out = self.feat3(out)

        out = self.hidden1(out)
        out = self.hidden2(out)
        out = self.hidden3(out)
        out = self.hidden4(out)
        out = self.hidden5(out)
        out = self.hidden6(out)
        out = self.hidden7(out)
        out = self.out(out)

        return out


'''

class NetRoT2(BaseModel):
    def __init__(self,in_shape_pose,cam_shape):
        super(NetRoT2,self).__init__()

        self.feat1 = basic_linear_rot2(in_shape_pose,128)
        self.feat2= basic_linear_rot2(6,64)
        self.feat3 = basic_linear_rot2(128, 256)

        self.hidden1 = basic_linear_rot2(256, 512)

        self.hidden2= basic_linear_rot2(512, 1024)

        self.hidden3= basic_linear_rot2(1024, 512)

        self.hidden4 = basic_linear_rot2(512, 256)

        self.hidden5=basic_linear_rot2(256, 128)

        self.hidden6 = basic_linear_rot2(128, 64)

        self.hidden7 = basic_linear_rot2(64, 30)

        self.out=nn.Linear(30,cam_shape)


    def forward(self, actpose,camseq):
        actpose=self.feat1(actpose)
        camseq=self.feat2(camseq)
        featall=actpose+camseq.repeat(1,2)
        input=self.feat3(featall)

        out=self.hidden1(input)
        out = self.hidden2(out)
        out = self.hidden3(out)
        out = self.hidden4(out)+input
        out = self.hidden5(out)+featall
        out = self.hidden6(out)+camseq
        out = self.hidden7(out)
        out = self.out(out)

        return out




class NetRoTOffset(BaseModel):
    def __init__(self,in_shape_pose,cam_shape):
        super(NetRoTOffset,self).__init__()

        self.feat1 = basic_linear_rot(in_shape_pose,128)
        self.feat2= basic_linear_rot(6,64)
        self.feat3 = basic_linear_rot(128, 256)

        self.hidden1 = basic_linear_rot(256, 512)

        self.hidden2= basic_linear_rot(512, 1024)

        self.hidden3= basic_linear_rot(1024, 512)

        self.hidden4 = basic_linear_rot(512, 256)

        self.hidden5=basic_linear_rot(256, 128)

        self.hidden6 = basic_linear_rot(128, 64)

        self.hidden7 = basic_linear_rot(64, 30)

        self.out=nn.Linear(30,cam_shape)


    def forward(self, actpose,camseq):
        actfeat=self.feat1(actpose)
        camfeat=self.feat2(camseq)
        camfeat=camfeat.repeat(1,2)
        out=actfeat+camfeat
        out=self.feat3(out)

        out=self.hidden1(out)
        out = self.hidden2(out)
        out = self.hidden3(out)
        out = self.hidden4(out)
        out = self.hidden5(out)
        out = self.hidden6(out)
        out = self.hidden7(out)
        out = self.out(out)

        return out


'''

if __name__ == '__main__':
    ####MFCC--(n_mfcc, t), t should be seq
    ###(batch,seq,featdim)
    ###downsampled MFCC+MFFCCdelta+orisig+onset(option)----+give GT beats
    insalfeat = torch.rand((64, 45, 6, 35))
    salen = SalEncoder(insalfeat.shape[3])
    # outsalfeat=salen(insalfeat)

    action = torch.rand((64, 45, 6, 35))
    inicam = torch.rand((64, 45, 6))
    emoint = torch.ones(64, 45, 1)
    emoint = emoint * 1.4

    # trajde=TrajDecoder(action.shape[0],action.shape[-1],inicam.shape[-1])
    # out=trajde(action,outsalfeat,emoint,inicam)
    g = Generator(insalfeat.shape[0], insalfeat.shape[-1], inicam.shape[-1])
    aaa = g(insalfeat, insalfeat, insalfeat, emoint, inicam)
    print(aaa)

    insalfeat = torch.randn((100, 90))
    outsal = torch.randn((100, 6))
    out = GRoT(insalfeat, outsal)
    print(out)
    print(outsal - out)
