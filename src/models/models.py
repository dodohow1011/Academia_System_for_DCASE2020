import torch.nn as nn
import torch
from dcase_util.data import ProbabilityEncoder
from utilities.utils import to_cuda_if_available
import numpy as np

def generate_label(pred):
    p = []
    thresh = [0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.45, 0.5, 0.35]
    for b in range(pred.size(0)):
        pred_bin = np.array([])
        for c in range(10):
            pred_b = ProbabilityEncoder().binarization(pred[b][c].unsqueeze(0).cpu().detach().numpy(),
                                                      binarization_type="global_threshold",
                                                      threshold=thresh[c])
            pred_bin = np.concatenate((pred_bin, pred_b), axis=0)
        p.append(pred_bin)

    p = torch.FloatTensor(p)
    p = to_cuda_if_available(p)
    return p

class PT(nn.Module):
    def __init__(self, n_in_channel, nclass=10, attention=False, activation="Relu", conv_dropout=0,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)]
                 ):
        super(PT, self).__init__()
        self.nclass = nclass
        self.nb_filters = nb_filters
        self.sigmoid = nn.Sigmoid()
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, 256)) for i in range(nclass)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, 1)) for i in range(nclass)])
        self.softmax = nn.Softmax(dim=1)
        self.attention = attention
        self.clf = nn.ModuleList([nn.Linear(256, 1) for i in range(nclass)])
        cnn = nn.Sequential()

        cnn.add_module('batchnorm', nn.BatchNorm2d(1, eps=0.001, momentum=0.99))
        
        def conv(i, batchNormalization=False, dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))
            if activ.lower() == "leakyrelu":
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module('relu{0}'.format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module('glu{0}'.format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module('cg{0}'.format(i), ContextGating(nOut))

        batch_norm = True
        # 128x862x64
        for i in range(len(nb_filters)):
            conv(i, batch_norm, conv_dropout, activ=activation)
            if (i+1) % 2 == 0:
                cnn.add_module('pooling{0}'.format(i//2), nn.MaxPool2d(pooling[i//2]))  # bs x tframe x mels
                cnn.add_module('dropout{0}'.format(i//2), nn.Dropout(conv_dropout))
        
        cnn.add_module('conv9', nn.Conv2d(128, 256, 1, 1, 0))
        self.cnn = cnn
    '''
    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def save(self, filename):
        torch.save(self.cnn.state_dict(), filename)
    '''

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.cnn(x)
        bs, chan, frames, freq = x.size()
        if freq != 1:
            warnings.warn(f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq")
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]
        
        weak = torch.FloatTensor([])
        weak = to_cuda_if_available(weak)
        if self.attention:
            for i in range(self.nclass):
                # attention weight
                sof = torch.sum(x * self.weights[i], dim=-1) + self.bias[i]  # [bs, frames, nclass]
                sof = sof / 256
                sof = self.softmax(sof)
            
                # contexual representation
                cr = torch.matmul(sof.unsqueeze(1), x).squeeze(1)

                # audio tagging
                at = self.clf[i](cr)
                at = nn.Sigmoid()(at)
                weak = torch.cat((weak, at), dim=-1)
            
        return weak

class PS(nn.Module):
    def __init__(self, n_in_channel, nclass=10, attention=False, activation="Relu", conv_dropout=0,
                 kernel_size=[3, 3, 3], padding=[1, 1, 1], stride=[1, 1, 1], nb_filters=[64, 64, 64],
                 pooling=[(1, 4), (1, 4), (1, 4)]
                 ):
        super(PS, self).__init__()
        # self.df = [46, 22, 92, 42, 82, 17, 13, 160, 74, 85]
        # self.df = [110, 70, 145, 75, 139, 84, 42, 160, 40, 129]
        # self.df = [137, 94, 134, 69, 132, 76, 34, 160, 30, 113]
        self.df = [160] * 10
        self.nclass = nclass
        self.nb_filters = nb_filters
        self.sigmoid = nn.Sigmoid()
        self.weights = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, self.df[i])) for i in range(nclass)])
        self.bias = nn.ParameterList([nn.Parameter(torch.zeros(1, 1)) for i in range(nclass)])
        self.softmax = nn.Softmax(dim=1)
        self.attention = attention
        self.clf = nn.ModuleList([nn.Linear(self.df[i], 1) for i in range(nclass)])
        cnn = nn.Sequential()
        
        cnn.add_module('batchnorm', nn.BatchNorm2d(1, eps=0.001, momentum=0.99))

        def conv(i, batchNormalization=False, dropout=None, activ="relu"):
            nIn = n_in_channel if i == 0 else nb_filters[i - 1]
            nOut = nb_filters[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, kernel_size[i], stride[i], padding[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut, eps=0.001, momentum=0.99))
            if activ.lower() == "leakyrelu":
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2))
            elif activ.lower() == "relu":
                cnn.add_module('relu{0}'.format(i), nn.ReLU())
            elif activ.lower() == "glu":
                cnn.add_module('glu{0}'.format(i), GLU(nOut))
            elif activ.lower() == "cg":
                cnn.add_module('cg{0}'.format(i), ContextGating(nOut))

        batch_norm = True
        # 128x862x64
        for i in range(len(nb_filters)):
            conv(i, batch_norm, conv_dropout, activ=activation)
            cnn.add_module('pooling{0}'.format(i), nn.MaxPool2d(pooling[i]))  # bs x tframe x mels
        
        self.cnn = cnn
    '''
    def load_state_dict(self, state_dict, strict=True):
        self.cnn.load_state_dict(state_dict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.cnn.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

    def save(self, filename):
        torch.save(self.cnn.state_dict(), filename)
    '''

    def forward(self, x):
        # input size : (batch_size, n_channels, n_frames, n_freq)
        # conv features
        x = self.cnn(x)

        bs, chan, frames, freq = x.size()
        if freq != 1:
            warnings.warn(f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq")
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]
        cnn_x = x

        weak = torch.FloatTensor([])
        weak = to_cuda_if_available(weak)
        strong = torch.FloatTensor([])
        strong = to_cuda_if_available(strong)

        # ATP
        if self.attention:
            for i in range(self.nclass):
                x_c = x[:, :, :self.df[i]]
                # attention weight
                sof = torch.sum(x_c * self.weights[i], dim=-1) + self.bias[i]  # [bs, frames, nclass]
                sed = self.sigmoid(sof).unsqueeze(-1)
                strong = torch.cat((strong, sed), dim=-1)
                sof = sof / self.df[i]
                sof = self.softmax(sof)
            
                # contexual representation
                cr = torch.matmul(sof.unsqueeze(1), x_c).squeeze(1)

                # audio tagging
                at = self.clf[i](cr)
                at = nn.Sigmoid()(at)
                weak = torch.cat((weak, at), dim=-1)

            phi = generate_label(weak).unsqueeze(1)
            strong = strong * phi

        return strong, weak, cnn_x
