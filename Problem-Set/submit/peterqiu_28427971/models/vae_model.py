from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, output_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_shape = output_shape

        self.base_size = (128, output_shape[1] // 8, output_shape[2] // 8)
        self.fc = nn.Linear(latent_dim, np.prod(self.base_size))
        self.deconvs = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_shape[0], 3, padding=1),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], *self.base_size)
        out = self.deconvs(out)
        return out

class ConvEncoder(nn.Module):
    def __init__(self, input_shape, latent_dim):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.convs = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
        )
        conv_out_dim = input_shape[1] // 8 * input_shape[2] // 8 * 256
        self.fc = nn.Linear(conv_out_dim, 2 * latent_dim)

    def forward(self, x):
        out = self.convs(x)
        out = out.view(out.shape[0], -1)
        mu, log_std = self.fc(out).chunk(2, dim=1)
        return mu, log_std

class VAEConvNet(nn.Module): 
    def __init__(self, input_shape, latent_size):
        """
        Simple VAE model. 
        
        Inputs: 
        - input_shape: size of input with shape (C, H, W)
        - latent_size: size of latent variable

        ------- Instruction -------
        You could follow the recommended network architecture in vae.ipynb
        
        ---------------------------
        """
        super().__init__()
        assert len(input_shape) == 3

        self.input_shape = input_shape
        self.latent_size = latent_size
        
        self.encoder = ConvEncoder(input_shape, latent_size)
        self.decoder = ConvDecoder(latent_size, input_shape)

    def forward(self, x):
        # TODO: finish the forward pass of VAE model
        # Normalize the input to [-1, 1]
        x = 2 * x - 1.0
        mu, log_std = self.encoder(x)
        std = torch.exp(0.5 * log_std)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_recon = torch.clamp(self.decoder(z), -1, 1) * 0.5 + 0.5

        return x_recon, mu, log_std

    def sample(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.latent_size).cuda()
            samples = torch.clamp(self.decoder(z), -1, 1)
        return samples.cpu().numpy() * 0.5 + 0.5


def vae_loss(x, x_recon, mu, log_std): 
    """
    Loss function for VAE
    Input: 
    - x: original image
    - x_recon: reconstructed image from VAE model
    - mu: mean vector of approximate posterior batch_size * J
    - log_std: variance vector in log space. 

    output: 
    - OrderedDict of the total loss, reconstruction loss and KL loss. 

    ------- Instruction -------
    Average the reconstruction loss and KL loss
    over batch dimension and sum over the feature dimension
    ---------------------------
    """
    # pass  # TODO: implement the loss for VAE model
    # print(x.shape)
    # print(x_recon.shape)
    # print(mu.shape)
    # input()
    # print(x_recon[0].unsqueeze(0).shape)
    # print(F.mse_loss(x_recon[0].unsqueeze(0),x[0].unsqueeze(0)))
    # recon_loss = F.mse_loss(x_recon, x)
    # print(x.mean())
    # print(x.max())
    # print(x.min())
    # print("Recon")
    # print(x_recon.mean())
    # print(x_recon.max())
    # print(x_recon.min())
    # input()
    # print(recon_loss)
    # print(torch.sum((x - x_recon)**2, dim = [1,2,3]))
    # print(torch.sum((x - x_recon)**2, dim = [1,2,3]).mean())
    # input()
    # print(log_std.shape)
    # print(log_std.max())
    # print(log_std.min())
    # print(log_std.mean())
    # print(log_std.exp().max())
    # print(log_std.exp().min())
    # print(log_std.exp().mean())
    # print(mu.max())
    # print(mu.min())
    # print(mu.mean())
    # input()
    # print()
    # print(1 + log_std - log_std.exp() - mu **2)
    # print(torch.sum(1 + log_std - log_std.exp() - mu **2 , dim = 1))
    # input()
    recon_loss = ((x - x_recon)**2).sum(dim = [1,2,3]).mean()
    kl_loss = torch.mean(-0.5 * torch.sum(1 + log_std - log_std.exp() - mu **2 , dim = 1))

    return OrderedDict(loss=recon_loss + kl_loss, recon_loss=recon_loss,
                        kl_loss=kl_loss)

