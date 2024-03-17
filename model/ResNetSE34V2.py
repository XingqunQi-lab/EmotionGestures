#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from model.ResNetBlocks import *
from model.utils import PreEmphasis
from model import vocab
import model.embedding_net

class ResNetSE(nn.Module):
    def __init__(self, args, block, layers, num_filters, nOut, z_obj, pose_level=3, encoder_type='SAP', n_mels=40, log_input=True, **kwargs):
        super(ResNetSE, self).__init__()

        print('Embedding size is %d, encoder %s.'%(nOut, encoder_type))

        self.pose_level = pose_level
        
        self.inplanes   = num_filters[0]
        # self.encoder_type = encoder_type
        # self.n_mels     = n_mels
        # self.log_input  = log_input
        self.z_obj = z_obj

        self.conv1 = nn.Conv2d(1, num_filters[0] , kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.conv_low = nn.Conv2d(64, 60, kernel_size=2, stride=1)
        self.bn_low = nn.BatchNorm2d(60)
        self.fc_low = nn.Linear(63 * 61, nOut)

        self.conv_mid = nn.Conv2d(32, 60, kernel_size=3, stride=1)
        self.bn_mid = nn.BatchNorm2d(60)
        self.fc_mid = nn.Linear(62 * 60, nOut)

        self.conv_high = nn.Conv2d(16, 60, kernel_size=3, stride=1)
        self.bn_high = nn.BatchNorm2d(60)
        self.fc_high = nn.Linear(62 * 62, nOut)

        self.layer1 = self._make_layer(block, num_filters[0], layers[0])
        self.layer2 = self._make_layer(block, num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(block, num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(block, num_filters[3], layers[3], stride=(2, 2))

        if self.z_obj:
            if isinstance(self.z_obj, vocab.Vocab):
                self.speaker_embedding = nn.Sequential(
                        nn.Embedding(z_obj.n_words, 16),
                        nn.Linear(16, 16)
                    )
                # self.speaker_mu = nn.Linear(16, 16)
                # self.speaker_logvar = nn.Linear(16, 16)
            # else:
            #     pass
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, self.pose_level * 3)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, x, vid_indices):

        # with torch.no_grad():
        #     with torch.cuda.amp.autocast(enabled=False):
        #         x = self.torchfb(x)+1e-6
        #         if self.log_input: x = x.log()
        #         x = self.instancenorm(x).unsqueeze(1)
        # print(x)
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.layer1(x)
        feat1 = self.layer2(x)
        feat2 = self.layer3(feat1)
        feat3 = self.layer4(feat2)
        #print('feat1 shape is: ', feat1.shape) # (32, 64, 64, 62)
        #print('feat2 shape is: ', feat2.shape) # (32, 128, 32, 31)
        #print('feat3 shape is: ', feat3.shape) # (32, 256, 16, 16)

        feat1 = self.conv_low(feat1)
        feat1 = self.relu(feat1)
        feat1 = self.bn_low(feat1)
        #print('feat1 shape is: ', feat1.shape) # (32, 64, 63, 61)
        #feat1 = feat1.reshape(batch_size, -1, feat1.shape[-1])
        #feat1 = feat1.transpose(1, 2)
        feat1 = feat1.reshape(batch_size, 60, -1)
        feat_low = self.fc_low(feat1)
        #feat_low = feat_low.reshape(batch_size, -1, feat_low.shape[-1])

        pixel_shuffle_mid = nn.PixelShuffle(2)
        feat2 = pixel_shuffle_mid(feat2)
        feat2 = self.conv_mid(feat2)
        feat2 = self.relu(feat2)
        feat2 = self.bn_mid(feat2)
        #print('feat2 shape is: ', feat2.shape) # (32, 128, 62, 60)
        #feat2 = feat2.reshape(batch_size, -1, feat2.shape[-1])
        #feat2 = feat2.transpose(1, 2)
        feat2 = feat2.reshape(batch_size, 60, -1)
        feat_mid = self.fc_mid(feat2)
        #feat_mid = feat_mid.reshape(batch_size, -1, feat_mid.shape[-1])

        pixel_shuffle_high = nn.PixelShuffle(4)
        feat3 = pixel_shuffle_high(feat3)
        feat3 = self.conv_high(feat3)
        feat3 = self.relu(feat3)
        feat3 = self.bn_high(feat3)
        #print('feat3 shape is: ', feat3.shape) # (32, 256, 16, 16)
        #feat3 = feat3.reshape(batch_size, -1, feat3.shape[-1])
        #feat3 = feat3.transpose(1, 2)
        feat3 = feat3.reshape(batch_size, 60, -1)
        feat_high = self.fc_high(feat3)
        #feat_high = feat_high.reshape(batch_size, -1, feat_high.shape[-1])

        linear_blend_feat = []
        weight = None

        if self.z_obj:
            if self.speaker_embedding:
                assert vid_indices is not None
                z_context = self.speaker_embedding(vid_indices)
                # z_mu = self.speaker_mu(z_context)
                # z_logvar = self.speaker_logvar(z_context)
                # z_context = model.embedding_net.reparameterize(z_mu, z_logvar)
            # else:
            #     z_mu = z_logvar = None
            #     z_context = torch.randn(batch_size, 16)

            x = F.elu(z_context)
            x = F.elu(self.fc1(x))
            x = self.fc2(x).reshape(batch_size, 3, self.pose_level)
            weight = F.softmax(x, dim=1)
            
            for i in range(self.pose_level):
                weight_low = weight[:, 0, i].unsqueeze(1).unsqueeze(1)
                weight_mid = weight[:, 1, i].unsqueeze(1).unsqueeze(1)
                weight_high = weight[:, 2, i].unsqueeze(1).unsqueeze(1)
                #print('feat_low shape is: ', feat_low.shape)
                #print('feat_mid shape is: ', feat_mid.shape)
                #print('feat_high shape is: ', feat_high.shape)
                temp = feat_low * weight_low + feat_mid * weight_mid + feat_high * weight_high
                linear_blend_feat.append(temp)
        else:
            linear_blend_feat = [None, None, None, None, None, None]
            z_mu = z_logvar = None
            z_context = None

        return weight, feat_low, feat_mid, feat_high, linear_blend_feat
        # return weight, feat_low, feat_mid, feat_high, linear_blend_feat, z_context, z_mu, z_logvar
