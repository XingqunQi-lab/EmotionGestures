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
        self.Encoder = nn.Sequential(nn.Linear(512, 512), # Only for single left hand or right hand 20 joints x 3 (angle-axis) = 60
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
             )
        
        self.Posterior_Y_embedding = nn.Sequential(
             #nn.Dropout(0.2),
            nn.Linear(8, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            #nn.Dropout(0.2),
            #nn.Linear(128, 64),
            #nn.Dropout(0.2),
            #nn.Linear(64, 32)
             )
        self.fc_mu = nn.Linear(60 * 32, 32)
        self.fc_var = nn.Linear(60 * 32, 32)
        
        self.Decoder = nn.Sequential(
             nn.Linear(32, 64),
             nn.Dropout(0.2),
             nn.Linear(64, 128),
             nn.Dropout(0.2),
             nn.Linear(128, 256),
             nn.Dropout(0.2),
             nn.Linear(256, 512),
             nn.Dropout(0.2),
             nn.Linear(512, 512),
             #nn.Dropout(0.2),
             #nn.Linear(128, 90)
             #nn.Dropout(0.5)
             )
        #self.noise = nn.Sequential(
             #nn.Linear(32, 128),
             #nn.Dropout(0.2),
             #nn.Linear(128, 512)
             #)


        self.fusion_z_posterior = nn.Sequential(
             nn.Linear(64, 60 * 32),
             nn.Dropout(0.2),
             nn.Linear(60 * 32, 60 * 32)
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
        latent_vector = latent_vector.reshape(x.shape[0], -1)
        mu = self.fc_mu(latent_vector)
        log_var = self.fc_var(latent_vector)
        z = self.reparameterize(mu, log_var)
        post_y = self.Posterior_Y_embedding(y)
        
        z = torch.cat([z, post_y], dim = 1)
        
        z = self.fusion_z_posterior(z)
        z = z.reshape(Input.shape[0], 60, 32)
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
        z = z.reshape(Input.shape[0], 60, 32)
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


class MLP_Reconstruct_v2(nn.Module):
    def __init__(self, bath = True):
        super(MLP_Reconstruct_v2, self).__init__()
        #self.batch = batch
        #self.mlploss = nn.L1Loss()
        #MLP_dims = [4096, 1024]
        self.Encoder = nn.Sequential(nn.Linear(512, 512), # Only for single left hand or right hand 20 joints x 3 (angle-axis) = 60
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
             )
        
        self.Posterior_Y_embedding = nn.Sequential(
             #nn.Dropout(0.2),
            nn.Linear(8, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            #nn.Dropout(0.2),
            #nn.Linear(128, 64),
            #nn.Dropout(0.2),
            #nn.Linear(64, 32)
             )
        self.fc_mu = nn.Sequential(
            nn.Linear(60 * 32, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
             )
        self.fc_var = nn.Sequential(
            nn.Linear(60 * 32, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
             )
        
        self.Decoder = nn.Sequential(
             nn.Linear(32, 64),
             nn.Dropout(0.2),
             nn.Linear(64, 128),
             nn.Dropout(0.2),
             nn.Linear(128, 256),
             nn.Dropout(0.2),
             nn.Linear(256, 512),
             nn.Dropout(0.2),
             nn.Linear(512, 512),
             #nn.Dropout(0.2),
             #nn.Linear(128, 90)
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
            nn.Linear(256, 512),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 60 * 32),
            #nn.Dropout(0.2),
            #nn.Linear(64, 32)
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
        latent_vector = latent_vector.reshape(x.shape[0], -1)
        mu = self.fc_mu(latent_vector)
        log_var = self.fc_var(latent_vector)
        z = self.reparameterize(mu, log_var)
        post_y = self.Posterior_Y_embedding(y)
        
        z = torch.cat([z, post_y], dim = 1)
        
        z = self.fusion_z_posterior(z)
        z = z.reshape(Input.shape[0], 60, 32)
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
        z = z.reshape(Input.shape[0], 60, 32)
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



class MLP_Reconstruct_v3(nn.Module):
    def __init__(self, bath = True):
        super(MLP_Reconstruct_v3, self).__init__()
        #self.batch = batch
        #self.mlploss = nn.L1Loss()
        #MLP_dims = [4096, 1024]
        self.Encoder = nn.Sequential(
            
            nn.Conv1d(60,32,3,padding=1),
            nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(32),
            nn.Conv1d(32,16,3,padding=1),
            nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(16), 
            nn.Conv1d(16, 8, 5,stride=2,padding=2),
            nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(8), 
            nn.Conv1d(8, 4, 5,stride=2,padding=2),
            nn.LeakyReLU(0.2, True),
			nn.BatchNorm1d(4),              
		)
        
        self.Posterior_Y_embedding = nn.Sequential(
             #nn.Dropout(0.2),
            nn.Linear(8, 16),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            #nn.Dropout(0.2),
            #nn.Linear(128, 64),
            #nn.Dropout(0.2),
            #nn.Linear(64, 32)
             )
        self.fc_mu = nn.Sequential(
            nn.Linear(4 * 128, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
             )
        self.fc_var = nn.Sequential(
            nn.Linear(4 * 128, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
             )
        
        self.Decoder = nn.Sequential(
            nn.ConvTranspose1d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(8),
            nn.ConvTranspose1d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(16),
            nn.Conv1d(16,32,3,padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(32),
            nn.Conv1d(32,60,3,padding=1),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(60),
            nn.Conv1d(60, 60, 3,padding=1),
             )
        #self.noise = nn.Sequential(
             #nn.Linear(32, 128),
             #nn.Dropout(0.2),
             #nn.Linear(128, 512)
             #)


        self.fusion_z_posterior = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 4 * 128),
            #nn.Dropout(0.2),
            #nn.Linear(512, 1024),
            #nn.Dropout(0.2),
            #nn.Linear(1024, 60 * 32),
            #nn.Dropout(0.2),
            #nn.Linear(64, 32)
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
        #print('latent_vector shape is: ', latent_vector.shape)
        latent_vector = latent_vector.reshape(x.shape[0], -1)
        mu = self.fc_mu(latent_vector)
        log_var = self.fc_var(latent_vector)
        z = self.reparameterize(mu, log_var)
        post_y = self.Posterior_Y_embedding(y)
        #print('post_y shape is: ', post_y.shape)
        
        z = torch.cat([z, post_y], dim = 1)
        
        z = self.fusion_z_posterior(z)
        z = z.reshape(Input.shape[0], 4, 128)
        output = self.Decoder(z)
        #print("output shape is: ", output.shape)

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
        z = z.reshape(n, 4, 128)
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