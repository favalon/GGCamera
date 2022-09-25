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

    def __init__(self, batch, inactfeat, incamfeat):
        super(TrajDecoder, self).__init__()

        self.actionbranch1 = basic_linear(inactfeat, 64)
        self.actmove = basic_linear(inactfeat, 64)
        self.maxpool = nn.MaxPool1d(2, stride=2)
        self.actionbranch2 = basic_linear(256, 256)
        self.actionbranch3 = basic_linear(256, 512)

        self.inicambranch1 = basic_linear(incamfeat, 64)
        self.inicambranch2 = basic_linear(128, 128)

        self.emobranch = basic_linear(128, 128)

        self.aes_scorebranch = basic_linear(128, 128)
        self.iniposbranch = basic_linear(6, 128)

        self.rnn1 = nn.GRU(bidirectional=False, hidden_size=512, input_size=512, num_layers=1, batch_first=True)
        self.h0 = nn.Parameter(torch.randn(1, batch, 512).type(T.FloatTensor), requires_grad=True)

        self.dropout = nn.Dropout(p=0.2)

        self.outbranch1 = basic_linear(512, 128)

        self.meanpool = nn.AvgPool1d(2, stride=2)

        self.outbranch2 = basic_linear(64, 64)

        self.fc = nn.Linear(64, incamfeat)

    def forward(self, action, act_v, salmask, salfeat, inipos, aes_score, emoint, inicam):
        # action(B,N,6,30), salfeat(B,N,6,64),
        # iniposB, N, 2, 3) aes_score(n, N, 1) emoint(B,N,1),inicam(B,N,6)

        action = self.actionbranch1(action)
        act_v = self.actmove(act_v)
        action = action + act_v

        action = action * salmask
        action = action.reshape((action.shape[0], action.shape[1], action.shape[2] * action.shape[3]))
        action = self.maxpool(action)

        inicam_1 = self.inicambranch1(inicam)
        action = torch.cat((action, inicam_1), dim=2)
        action = self.actionbranch2(action)

        inicam_2 = self.inicambranch2(
            torch.cat((inicam_1, torch.randn(action.shape[0], action.shape[1], 64).type_as(inicam_1)), dim=2))
        emofeat = torch.cat(
            (emoint.repeat((1, 1, 64)), torch.randn(action.shape[0], action.shape[1], 64).type_as(emoint)), dim=2)
        emofeat = self.emobranch(emofeat)

        scorefeat = torch.cat(
            (emoint.repeat((1, 1, 64)), torch.randn(action.shape[0], action.shape[1], 64).type_as(aes_score)), dim=2)
        scorefeate = self.aes_scorebranch(scorefeat)

        inipos = self.iniposbranch(
            inipos.reshape((inipos.shape[0], inipos.shape[1], inipos.shape[2] * inipos.shape[3])))

        action = action + torch.cat((inicam_2, emofeat), dim=2)

        action = self.actionbranch3(action)
        action = action + salfeat

        action = action * emoint

        out, h1 = self.rnn1(action, self.h0)

        out = self.dropout(out)

        out = out * emoint

        out = self.outbranch1(out)
        out = out + inicam_2
        out = self.meanpool(out)
        out = out * emoint

        out = self.outbranch2(out)
        out = out + inicam_1
        out = self.fc(out)

        return out.contiguous()


class Generator(BaseModel):
    def __init__(self, batch, in_shape, out_shape):
        super(Generator, self).__init__()

        self.sal_encoder = SalEncoder(in_shape)
        self.tj_decoder = TrajDecoder(batch, in_shape, out_shape)

    def forward(self, act_diff, action, act_v, inipos, aes_score, emoint, inicam):
        outmask, outfeat = self.sal_encoder(act_diff)
        output = self.tj_decoder(action, act_v, outmask, outfeat, inipos, aes_score, emoint, inicam)
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
