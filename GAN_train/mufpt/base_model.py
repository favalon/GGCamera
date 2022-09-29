import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

T = torch
if torch.cuda.is_available():
    T = torch.cuda


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.layers = {}

    def init_DNN(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.in_features
                y = 1.0 / np.sqrt(n)
                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)

    def init_from_caffe(self, weights):
        for k in self.layers.keys():
            self.layers[k].weight.data = torch.from_numpy(weights[k][0])
            self.layers[k].bias.data = torch.from_numpy(weights[k][1])

    def initialize_kernel(self, kernel_initializer):
        # for k in self.layers.keys():
        #    kernel_initializer(self.layers[k].weight)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kernel_initializer(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)

    def initialize_bias(self, bias_initializer):
        # for k in self.layers.keys():
        #    bias_initializer(self.layers[k].bias)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    bias_initializer(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_weigths(self):
        list_w = []
        list_b = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                list_w.append(m.weight.data)

                if m.bias is not None:
                    list_b.append(m.bias.data)
                else:
                    list_b.append(0)

        return list_w, list_b

    def set_weigths(self, list_w, list_b):
        i = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data[:] = list_w[i][:]

                if m.bias is not None:
                    m.bias.data[:] = list_b[i][:]

                i += 1

    def save(self, path):
        state = self.state_dict()
        torch.save(state, path)

    def load(self, path):
        if (torch.cuda.is_available()):

            state = torch.load(path)
            self.load_state_dict(state)
        else:
            state = torch.load(path, map_location=lambda storage, loc: storage)
            self.load_state_dict(state)


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

        self.softmax = nn.Softmax(dim=0)

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

        self.rnn1 = nn.GRU(bidirectional=False, hidden_size=512, input_size=512, num_layers=1, batch_first=True)
        self.h0 = nn.Parameter(torch.randn(1, batch, 512).type(T.FloatTensor), requires_grad=True)

        self.dropout = nn.Dropout(p=0.2)

        self.outbranch1 = basic_linear(512, 128)

        self.meanpool = nn.AvgPool1d(2, stride=2)

        self.outbranch2 = basic_linear(64, 64)

        self.fc = nn.Linear(64, incamfeat)

    def forward(self, action, act_v, salmask, salfeat, emoint, inicam, rands):
        ###### action(B,N,6,30), salfeat(B,N,6,64),emoint(B,N,1),inicam(B,N,6)

        action = self.actionbranch1(action)
        act_v = self.actmove(act_v)
        action = action + act_v

        action = action * salmask
        action = action.reshape((action.shape[0], action.shape[1], action.shape[2] * action.shape[3]))
        action = self.maxpool(action)

        inicam_1 = self.inicambranch1(inicam)
        action = torch.cat((action, inicam_1), dim=2)
        action = self.actionbranch2(action)

        inicam_2 = self.inicambranch2(torch.cat((inicam_1, rands[1].type_as(inicam_1)), dim=2))
        emofeat = torch.cat((emoint.repeat((1, 1, 64)), rands[2].type_as(emoint)), dim=2)
        emofeat = self.emobranch(emofeat)

        action = action + torch.cat((inicam_2, emofeat), dim=2)

        action = self.actionbranch3(action)
        action = action + salfeat

        action = action * emoint

        out, h1 = self.rnn1(action, self.h0)

        # out = self.dropout(out)

        # print(out[0, 0, 0])

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

    def forward(self, act_diff, action, act_v, emoint, inicam, rands):
        outmask, outfeat = self.sal_encoder(act_diff)
        output = self.tj_decoder(action, act_v, outmask, outfeat, emoint, inicam, rands)
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


def shakiness_check_t_adv_sm(camera_tj):
    ################camera_tj:(B,N,2,3)

    #########seq(B,N,2,3)

    #############processing smoothness
    camera_tj = camera_tj.reshape(camera_tj.shape[0], camera_tj.shape[1], camera_tj.shape[2] * camera_tj.shape[3])
    filters = torch.ones(1, 1, 3, 1) / 3
    filters = filters.type_as(camera_tj)
    filtn = 1
    camera_tj_clone = camera_tj.clone()
    for i in range(camera_tj.shape[0]):
        for j in range(camera_tj.shape[2]):
            temp = camera_tj_clone[i, :, j].unsqueeze(0).unsqueeze(0).unsqueeze(3)
            tj_smooth = F.conv2d(temp, filters, padding=0)
            camera_tj_clone[i, filtn:-filtn, j] = tj_smooth[0, 0, :, 0]

    camera_tj = camera_tj_clone
    camera_tj = camera_tj.reshape(camera_tj.shape[0], camera_tj.shape[1], 2, 3)

    st_prev = camera_tj[:, :-2, :, :]
    st_central = camera_tj[:, 1:-1, :, :]
    st_next = camera_tj[:, 2:, :, :]

    spd_prev = st_central - st_prev
    spd_next = st_next - st_central

    sign_prev = torch.sign(spd_prev)
    sign_next = torch.sign(spd_next)

    ####peak/valley

    sign_central = sign_next * sign_prev * (-1)
    sign_central = sign_central > 0
    sign_central = sign_central.int()

    ###back to real index
    pads = torch.zeros(camera_tj.shape[0], 1, camera_tj.shape[2], camera_tj.shape[3]).type_as(camera_tj).int()
    motion_beat = torch.cat((pads, sign_central), dim=1)
    motion_beat = motion_beat.reshape(motion_beat.shape[0], motion_beat.shape[1],
                                      motion_beat.shape[2] * motion_beat.shape[3])

    camera_tj = camera_tj.reshape(camera_tj.shape[0], camera_tj.shape[1], camera_tj.shape[2] * camera_tj.shape[3])

    ###get beat index array from (B,N,6)

    measurement = torch.zeros(camera_tj.shape[0], camera_tj.shape[2]).type_as(camera_tj)
    for i in range(motion_beat.shape[0]):
        for j in range(motion_beat.shape[2]):

            ##########(N,1)
            beats = torch.nonzero(motion_beat[i, :, j])[:, 0]
            seq = camera_tj[i, :, j]

            if (len(beats) == 0):
                measurement[i, j] = 0
            else:

                temp = torch.zeros(len(beats) + 2).type_as(camera_tj).long()
                temp[1:-1] = beats
                temp[0] = 0
                temp[-1] = len(seq) - 1

                value = torch.take(seq, temp)

                vldiff = torch.abs(value[1:-1] - value[:-2])
                vrdiff = torch.abs(value[1:-1] - value[2:])

                ldiff = torch.abs(temp[1:-1] - temp[:-2])
                rdiff = torch.abs(temp[1:-1] - temp[2:])

                interval = ldiff + rdiff
                log_vldiff = vldiff <= vrdiff
                log_vrdiff = vrdiff < vldiff

                diff = log_vldiff.float() * vldiff + log_vrdiff.float() * vrdiff

                measurement[i, j] = torch.sum(diff / interval.float())

    return measurement
