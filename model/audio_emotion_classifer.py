# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
#from .audio_emotion_classifer_submodules import *
import torchvision.models as models
import functools
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from .emotion_ResNetSE34V2 import *
from .emotion_ResNetBlocks import *
#from convolutional_rnn import Conv2dGRU

class EmotionNet(nn.Module):
    def __init__(self):
        super(EmotionNet, self).__init__()
        num_filters = [32, 64, 128, 256]
        
        self.emotion_encoder = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters)
        
        self.emotion_eocder_fc = nn.Sequential(
            nn.Linear(256 * 16 * 16, 4096),
            nn.ReLU(True),
            nn.Linear(4096,2048),
            nn.ReLU(True),
            nn.Linear(2048,512),
            nn.ReLU(True),
            nn.Linear(512,128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            )
        self.last_fc = nn.Linear(64,8)
        self.acn = nn.Softmax(dim=1)
    def forward(self, mfcc):
        mfcc= mfcc.unsqueeze(1)
        #print('mfcc shape is: ', mfcc.shape)
        #mfcc=torch.transpose(mfcc,2,3)
        feature = self.emotion_encoder(mfcc) ## (B, 256, 16, 16)
        # print(feature.shape)
        feature = feature.view(feature.size(0),-1)
        x = self.emotion_eocder_fc(feature)
        re = self.last_fc(x)
        # re= self.acn(re)

        return re

class DisNet(nn.Module):
    def __init__(self):
        super(DisNet, self).__init__()


        self.dis_fc = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,16),
            nn.ReLU(True),
            nn.Linear(16,1),
            nn.ReLU(True)
            )


    def forward(self, feature):

        re = self.dis_fc(feature)

        return re

