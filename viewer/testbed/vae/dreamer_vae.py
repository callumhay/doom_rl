import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn


class ObsDecoder(nn.Module):
    def __init__(self, output_shape, embed_size, info):
        """
        :param output_shape: tuple containing shape of output obs
        :param embed_size: the size of input vector, for dreamerv2 : modelstate 
        """
        super(ObsDecoder, self).__init__()
        c, h, w = output_shape
        activation = info['activation']
        d = info['depth']
        k  = info['kernel']
        conv1_shape = conv_out_shape(output_shape[1:], 0, k, 1)
        conv2_shape = conv_out_shape(conv1_shape, 0, k, 1)
        conv3_shape = conv_out_shape(conv2_shape, 0, k, 1)
        self.conv_shape = (4*d, *conv3_shape)
        self.output_shape = output_shape
        if embed_size == np.prod(self.conv_shape).item():
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(embed_size, np.prod(self.conv_shape).item())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4*d, 2*d, k, 1),
            activation(),
            nn.ConvTranspose2d(2*d, d, k, 1),
            activation(),
            nn.ConvTranspose2d(d, c, k, 1),
        )

    def forward(self, x):
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, embed_size)
        x = self.linear(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        x = self.decoder(x)
        mean = torch.reshape(x, (*batch_shape, *self.output_shape))
        obs_dist = td.Independent(td.Normal(mean, 1), len(self.output_shape))
        return obs_dist
    
def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)

def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))