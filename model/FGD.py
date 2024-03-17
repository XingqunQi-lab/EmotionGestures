#Author: xingqun qi
import os
import sys
import numpy as np
import scipy.io as io

rng = np.random.RandomState(23456)

import torch
import torchvision
from torch import nn
#from torch.autograd import Variable
#from torch.utils.data import DataLoader
#from torchvision.utils import save_image
#from torchvision.datasets import MNIST
#import os
#from PIL import Image,ImageDraw 
#from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

#multi_gpu = True
#BatchNorm = SynchronizedBatchNorm2d if multi_gpu else nn.BatchNorm2d

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP_Reconstruct(nn.Module):
    def __init__(self, bath = True):
        super(MLP_Reconstruct, self).__init__()
        #self.batch = batch
        #self.mlploss = nn.L1Loss()
        #MLP_dims = [4096, 1024]
        self.Encoder = nn.Sequential(nn.Linear(282, 512), # Only for single left hand or right hand 20 joints x 3 (angle-axis) = 60
             nn.Dropout(0.2),
             nn.Linear(512, 512),
             nn.Dropout(0.2),
             nn.Linear(512, 512),
             #nn.Dropout(0.2),
             #nn.Linear(256, 256),
             #nn.Dropout(0.2),
             #nn.Linear(256, 512)
             )
        """
        self.IDEmbedding = nn.Sequential(
             nn.Linear(4, 32),
             #nn.Dropout(0.5),
             #nn.Linear(64, 64),
             #nn.Dropout(0.5),
             #nn.Linear(64, 32),
             #nn.Dropout(0.5)
             #nn.Linear(1, 512)
             )
        """
        self.Decoder = nn.Sequential(
             nn.Linear(512, 512),
             nn.Dropout(0.2),
             nn.Linear(512, 512),
             nn.Dropout(0.2),
             nn.Linear(512, 282),
             #nn.Dropout(0.2),
             #nn.Linear(128, 128),
             #nn.Dropout(0.2),
             #nn.Linear(128, 90)
             #nn.Dropout(0.5)
             )

    def forward(self, Input):
        # set the image to the teacher to obtain the teacher_latent_vector, Reconstruction features
        #B, C, N = Input.shape[0], Input.shape[1], Input.shape[2]
        #ID = ID.transpose(1, 2)
        #ID = ID[:, 0, :]
        #ID[:,0] = 1
        #ID[:,1] = 0        
        #ID_embedding = self.IDEmbedding(ID)
        #print('ID_embedding shape is:',ID_embedding.shape)
        x = Input
        
        latent_vector = self.Encoder(x)
        #latent_vector = latent_vector #+ ID_embedding
        #print('latent_vector shape is: ', latent_vector.shape)
        output = self.Decoder(latent_vector)
        #output = output.reshape(B, C, N)
        return output, latent_vector
    
    """
    def PHAE_encoder(self, Input):
        x = Input
        
        latent_vector = self.Encoder(x)
        return latent_vector
    
    def PHAE_decoder(self, middle):
        
        x = middle
        output = self.Decoder(x)
        
        return output
    """