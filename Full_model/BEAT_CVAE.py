#Author: xingqun qi
import os
import sys
import numpy as np
import scipy.io as io
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
#rng = np.random.RandomState(23456)

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
#from torchvision.datasets import MNIST
import os
from PIL import Image,ImageDraw 









class MLP_Reconstruct(nn.Module):
    def __init__(self, bath = True):
        super(MLP_Reconstruct, self).__init__()
        #self.batch = batch
        #self.mlploss = nn.L1Loss()
        #MLP_dims = [4096, 1024]
        self.Encoder = nn.Sequential(nn.Linear(90, 128), # Only for single left hand or right hand 20 joints x 3 (angle-axis) = 60
             nn.Dropout(0.2),
             nn.Linear(128, 128),
             nn.Dropout(0.2),
             nn.Linear(128, 256),
             nn.Dropout(0.2),
             nn.Linear(256, 256),
             nn.Dropout(0.2),
             nn.Linear(256, 512)
             )
        
        self.Posterior_Y_embedding = nn.Sequential(
             #nn.Dropout(0.2),
             nn.Linear(90, 64),
             nn.Dropout(0.2),
             nn.Linear(64, 32),
             #nn.Dropout(0.2),
             #nn.Linear(256, 256),
             #nn.Dropout(0.2),
             #nn.Linear(256, 512)
             )
        self.fc_mu = nn.Linear(512, 32)
        self.fc_var = nn.Linear(512, 32)
        
        self.Decoder = nn.Sequential(
             nn.Linear(512, 256),
             nn.Dropout(0.2),
             nn.Linear(256, 256),
             nn.Dropout(0.2),
             nn.Linear(256, 128),
             nn.Dropout(0.2),
             nn.Linear(128, 128),
             nn.Dropout(0.2),
             nn.Linear(128, 90)
             #nn.Dropout(0.5)
             )
        #self.noise = nn.Sequential(
             #nn.Linear(32, 128),
             #nn.Dropout(0.2),
             #nn.Linear(128, 512)
             #)


        self.fusion_z_posterior = nn.Sequential(
             nn.Linear(64, 256),
             nn.Dropout(0.2),
             nn.Linear(256, 512)
             )
    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu             

    def sample_p_0(n): # n = args.batchsize, sig=args.e_init_sig
            return 1.0 * torch.randn(*[n, 32])#.to(device)
    def forward(self, Input, y):
        # set the image to the teacher to obtain the teacher_latent_vector, Reconstruction features

        x = Input
        
        latent_vector = self.Encoder(x)
        mu = self.fc_mu(latent_vector)
        log_var = self.fc_var(latent_vector)
        z = self.reparameterize(mu, log_var)
        post_y = self.Posterior_Y_embedding(y)
        
        z = torch.cat([z, post_y], dim = 1)
        
        z = self.fusion_z_posterior(z)
        output = self.Decoder(z)

        return output, mu, log_var


    def sample(self, y):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        #y = kwargs['labels'].float()
        #z = self.sample_p_0(n=y.shape[0],sig=1.0)

        #print('y.shape[0] is:', y.shape[0])
        n = y.shape[0]
        post_y = self.Posterior_Y_embedding(y)
        z = torch.randn(*[n, 32])
        z = z.to(post_y.device)
        z = torch.cat([z, post_y], dim=1)
        z = self.fusion_z_posterior(z)
        samples = self.Decoder(z)
        return samples   
    
    def VAE_encoder(self, Input):
        x = Input
        
        latent_vector = self.Encoder(x)
        return latent_vector
    
    def VAE_decoder(self, middle):
        
        x = middle
        output = self.Decoder(x)
        
        return output


