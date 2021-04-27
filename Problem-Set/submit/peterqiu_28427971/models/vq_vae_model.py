from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return x + self.net(x)


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x_shape = x.shape
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class MaskConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        assert mask_type == 'A' or mask_type == 'B'
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

    def forward(self, input):
        # print(input.shape)
        out = F.conv2d(input, self.weight * self.mask, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        return out

    def create_mask(self, mask_type):
        # ----------------- TODO ------------------ #
        # Implement Mask for type A and B layer here
        # ----------------------------------------- #
        # print(self.weight.shape)
        # input()
        self.mask = torch.ones_like(self.weight)

        # mask next pixels
        center_row = self.kernel_size[0] // 2
        center_col = self.kernel_size[1] // 2
        self.mask[:, :, center_row+1:, :] = 0
        self.mask[:, :, center_row, center_col+1:] = 0

        # mask channels for center pixels
        if mask_type == 'A':
            self.mask[:, :, center_row, center_col] = 0


class PixelCNNResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.ModuleList([
            LayerNorm(dim),
            nn.ReLU(),
            MaskConv2d('B', dim, dim // 2, 1),
            LayerNorm(dim // 2),
            nn.ReLU(),
            MaskConv2d('B', dim // 2, dim // 2, 3, padding=1),
            LayerNorm(dim // 2),
            nn.ReLU(),
            MaskConv2d('B', dim // 2, dim, 1)
        ])

    def forward(self, x):
        out = x
        for layer in self.block:
            out = layer(out)
        return x + out


class PixelCNN(nn.Module):
    def __init__(self, input_shape, code_size, dim=256, n_layers=7):
        """
        PixelCNN model. 

        Inputs: 
        - input_shape: (H, W) of the height abd width of the input
        - code_size: dimention of embedding vector in the codebook
        - dim: number of filters
        - n_layers: number of repeated block
        """
        super().__init__()
        # Since the input is a 2D grid of discrete values,
        # we'll have an input (learned) embedding layer to map the discrete values to embeddings

        self.embedding = nn.Embedding(code_size, dim)
        model = nn.ModuleList([MaskConv2d('A', dim, dim, 7, padding=3),
                               LayerNorm(dim), nn.ReLU()])
        for _ in range(n_layers - 1):
            model.append(PixelCNNResBlock(dim))
        model.extend([LayerNorm(dim), nn.ReLU(), MaskConv2d('B', dim, 512, 1),
                      nn.ReLU(), MaskConv2d('B', 512, code_size, 1)])
        self.net = model
        self.input_shape = input_shape
        self.code_size = code_size

    def forward(self, x):
        # print(x.shape)
        out = self.embedding(x).permute(0, 3, 1, 2).contiguous()
        # print(out.shape)
        # input()
        for layer in self.net:
            out = layer(out)
        return out

    def loss(self, x):
        # --------------- TODO: ------------------
        # Implement the loss function for PixelCNN
        # ----------------------------------------
        # x = x.to(dtype=torch.long)
        # x.cuda().contiguous()
        # print(x.shape)
        # print(self.net)
        out = self.forward(x)
        loss = F.cross_entropy(out, x)
        return OrderedDict(loss=loss)

    def sample(self, n):
        # ------ TODO: sample from the model --------------
        # Instruction:
        # Note that the generation process should proceed row by row and pixel by pixel.
        # *hint: use torch.multinomial for sampling
        # -------------------------------------------------
        samples = torch.zeros(n, *self.input_shape).long().cuda()
        for y in range(self.input_shape[0]):
            for x in range(self.input_shape[1]):
                # print(samples.dtype)
                logits = self.forward(samples)
                probs = F.softmax(logits, dim=1)
                sample = torch.multinomial(
                    probs[:, :, y, x], num_samples=1).squeeze()
                samples[:, y, x] = sample
                # print(samples)
                # input()
        return samples


class Quantize(nn.Module):
    """
    Vector quantisation. 

    Inputs: 
        - size: number of embedding vector in the codebook
        - code_dim: dimention of embedding vector in the codebook
    """

    def __init__(self, size, code_dim):
        super().__init__()
        # We use nn.Embedding to store embedding vectors.
        self.embedding = nn.Embedding(size, code_dim)
        self.embedding.weight.data.uniform_(-1./size, 1./size)

        self.code_dim = code_dim
        self.size = size

    def forward(self, z):
        # -------------------- TODO --------------------
        # Look at section 3.1 of the paper: Neural Discrete Representation Learning
        # , and finish the vector quantisation process.
        # Note: we have taken care of the straight-through estimator,
        #       note how we achieve it by using *detach* in PyTorch
        #
        # 1. Compute the distance between every pair of latent embedding and output features of the encoder
        # 2. Get the encoder input by finding nearest latent embeddings (Eq. (1) and (2) in the paper)
        # 3. The encoding indices is the indice of the retrieved latent embeddings
        # ----------------------------------------------
        # print(z.shape) [B x code_dim x H x W]
        # print(self.embedding.weight.shape)
        # print(self.size)
        # print(self.code_dim)
        # input()
        # [B x code_dim x H x W] -> [B x H x W x code_dim]
        z_trans = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z_trans.view(-1, self.code_dim)  # [BHW x code_dim]

        # 1. Compute Eculidean Distance # [BHW x size]
        dist = torch.sum(z_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight.t() ** 2, dim=0, keepdim=True) - \
            2 * torch.matmul(z_flat, self.embedding.weight.t())

        # 2&3. Find nearest latent embeddings and get quantized results
        # encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1) # [BHW x 1]
        encoding_indices = torch.argmin(dist, dim=1).view(
            z.shape[0], z.shape[2], z.shape[3])  # [B x H x W]
        quantized = self.embedding(encoding_indices)  # [B x H x W x code_dim]
        quantized = quantized.permute(
            0, 3, 1, 2).contiguous()  # [B x code_dim x H x W]
        # quantized = quantized.view(z.shape) # WRONG, use permute rather than view

        # encoding_one_hot = torch.zeros(encoding_indices.shape[0], self.size, device=z.device) # [BHW x size]
        # encoding_one_hot.scatter_(1, encoding_indices, 1)

        # quantized = torch.matmul(encoding_one_hot, self.embedding.weight) # [BHW x code_dim]
        # quantized = quantized.view(z.shape) # [B x code_dim x H x W]
        # print(quantized.max())
        # print(quantized.mean())
        # print(quantized.min())
        # input()
        # encoding_indices = encoding_indices.view(z.shape[0], z.shape[2], z.shape[3]) # [B x H x W]
        # print(encoding_indices.shape)
        return quantized, (quantized - z).detach() + z, encoding_indices


class VQVAENet(nn.Module):
    def __init__(self, code_dim, code_size):
        """
        VQ-VAE model. 

        Inputs: 
        - code_dim: dimention of embedding vector in the codebook
        - code_size: number of embedding vector in the codebook

        ------- Instruction -------
        - Build a codebook follow the instructions in *Quantize* class
        ---------------------------
        """
        super().__init__()
        self.code_size = code_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 4, stride=2, padding=1),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.codebook = Quantize(code_size, code_dim)

        self.decoder = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def encode_code(self, x):
        with torch.no_grad():
            x = 2 * x - 1
            z = self.encoder(x)
            indices = self.codebook(z)[2]
            return indices

    def decode_code(self, latents):
        with torch.no_grad():
            latents = self.codebook.embedding(
                latents).permute(0, 3, 1, 2).contiguous()
            return self.decoder(latents).permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5

    def forward(self, x):
        # x_tilde is the reconstructed images
        # diff1 and diff2 follow the last two terms of training objective (see Eq.(3) in the paper: Neural Discrete Representation Learning)
        # , where beta is set to 1.0

        x = 2 * x - 1
        z = self.encoder(x)
        e, e_st, _ = self.codebook(z)
        x_tilde = self.decoder(e_st)  # [-1, 1]
        # print(z.max())
        # print(z.mean())
        # print(z.min())
        # print(e.max())
        # print(e.mean())
        # print(e.min())
        # print(x.max())
        # print(x.mean())
        # print(x.min())
        # print(x_tilde.max())
        # print(x_tilde.mean())
        # print(x_tilde.min())
        # input()
        diff1 = torch.mean((z - e.detach()) ** 2)
        diff2 = torch.mean((e - z.detach()) ** 2)
        # diff1 = F.mse_loss(z, e.detach())
        # diff2 = F.mse_loss(e, z.detach())
        # print(diff1)
        # print(diff2)
        # print(diff1+diff2)
        # input()
        return x_tilde, diff1 + diff2


def vq_vae_loss(x_tilde, diff, x):
    # -------------- TODO --------------
    # finish the loss for VQ-VAE model
    # ----------------------------------
    # recon_loss = torch.mean((x_tilde - x) ** 2)
    # x: [0, 1]

    # print(x.max())
    # print(x.min())
    # print(x.mean())
    # print(x_tilde.max())
    # print(x_tilde.min())
    # print(x_tilde.mean())
    # input()
    recon_loss = F.mse_loss(x_tilde, x*2-1)
    reg_loss = diff
    loss = recon_loss + reg_loss
    return OrderedDict(loss=loss, recon_loss=recon_loss, reg_loss=reg_loss)
