import math

import torch
import torch.nn as nn
import torch.distributions as td
#import numpy as np

from conv_encode_decode import ConvEncoder, ConvDecoder

_LATENT_SPACE_SIZE = 2048

class DoomVAE(nn.Module):
  def __init__(self, input_shape) -> None:
    super(DoomVAE, self).__init__()
    
    self.input_shape = input_shape
    self.latent_dim = _LATENT_SPACE_SIZE
    self.running_mse = None
    
    # Encoder
    self.conv_encoder = ConvEncoder(input_shape)
    
    # Latent representation 
    prelatent_size = self.conv_encoder.conv_output_size
    self.z_mu      = nn.Linear(prelatent_size, self.latent_dim)
    self.z_logvar   = nn.Linear(prelatent_size, self.latent_dim)
    
    # This is very important, apparently negative logvar values can explode the gradient
    nn.init.constant_(self.z_logvar.weight, 0.0)
    nn.init.constant_(self.z_logvar.bias, 0.0)
    
    # Decoder
    self.decoder_input = nn.Linear(self.latent_dim, prelatent_size)
    self.conv_decoder = ConvDecoder(self.input_shape, self.conv_encoder.output_sizes)
    
    
  def encode(self, input:torch.Tensor) -> list[torch.Tensor]:
    """
    Encodes the input by passing it through the encoder network,
    resulting in the latent representation for it.
    Args:
        input (torch.Tensor): Input tensor to the encoder [B x C x H x W]
    Returns:
        list[torch.Tensor]: List of latent representations [mu, std_dev]
    """
    result = self.conv_encoder(input)
    
    # Split the result of the flattened convolutional encoding into mu (mean) 
    # and sigma (standard deviation) of the latent Gaussian distribution
    mu = self.z_mu(result)
    logvar = self.z_logvar(result)
    return [mu, logvar]
  
  def decode(self, z:torch.Tensor) -> torch.Tensor:
    """
    Maps the given latent representation back into image space.
    Args:
        z (torch.Tensor): [B x D], if mu and sigma values are present then
        you must call 'reparameterize' in order to convert them to z.
    Returns:
        torch.Tensor: [B x C x H x W] tensor representing the image.
    """
    conv_ch, conv_h, conv_w = self.conv_encoder.conv_output_shape
    result = self.decoder_input(z)
    result = result.view(-1, conv_ch, conv_h, conv_w)
    return self.conv_decoder(result)
    
  def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization to sample from a Gaussian distribution N(mu,sigma) via a random sample in N(0,1).
    Args:
        mu (torch.Tensor): The mean of the latent Gaussian [B x D]
        sigma (torch.Tensor): The standard deviation of the latent Gaussian [B x D]
    Returns:
        torch.Tensor: 'z' latent representation as a sampled tensor [B x D]
    """
    std = torch.exp(0.5 * logvar)
    #std = torch.nn.functional.softplus(std) + 0.1
    return torch.randn_like(std) * std + mu
  
  def forward(self, input:torch.Tensor) -> list[torch.Tensor]:
    mu, logvar = self.encode(input)
    z = self.reparameterize(mu, logvar)
    return [self.decode(z), input, mu, logvar]

  def loss_function(self, reconstruction, input, mu, logvar, kl_beta=0.8):

    # Normal distribution logprob loss
    sigma = ((input-reconstruction)**2).mean([0,1,2,3], keepdim=True).sqrt()
    reconst_dist = td.Independent(td.Normal(reconstruction, sigma), len(self.input_shape))
    reconst_loss = -torch.mean(reconst_dist.log_prob(input))
    
    '''
    prior_dist = self.RSSM.get_dist(prior)
    post_dist = self.RSSM.get_dist(posterior)
    alpha = 0.8
    kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self.RSSM.get_dist(self.RSSM.rssm_detach(posterior)), prior_dist))
    kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(post_dist, self.RSSM.get_dist(self.RSSM.rssm_detach(prior))))
    if self.kl_info['use_free_nats']:
        free_nats = self.kl_info['free_nats']
        kl_lhs = torch.max(kl_lhs,kl_lhs.new_full(kl_lhs.size(), free_nats))
        kl_rhs = torch.max(kl_rhs,kl_rhs.new_full(kl_rhs.size(), free_nats))
    kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs
    '''
    
    # Reconst MSE Mean
    #reconst_loss = nn.functional.mse_loss(reconstruction, input.detach())
    # KLD Mean
    kld_loss = torch.mean(-0.5 * torch.sum(1.0 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)

    return [reconst_loss + kl_beta * kld_loss, reconst_loss.detach(), kld_loss.detach()]
    

  def sample(self, num_samples:int, device:torch.device) -> torch.Tensor:
    """
    Samples from the latent space and returns the corresponding image space map.
    Args:
        num_samples (int): The number of samples
        device (torch.device): The device running the model
    Returns:
        torch.Tensor: Resulting image space map [num_samples x C x H x W]
    """
    z = torch.randn(num_samples, self.latent_dim).to(device)
    return self.decode(z)
  
  def sample_mean(self, input:torch.Tensor) -> torch.Tensor:
    mu, _ = self.encode(input)
    return self.decode(mu)
  
  def generate(self, x:torch.Tensor) -> torch.Tensor:
    """
    Given an input image, returns the reconstructed image.
    Args:
        x (torch.Tensor): The input image tensor [B x C x H x W]
    Returns:
        torch.Tensor: The reconstructed image tensor [B x C x H x W]
    """
    return self.forward(x)[0]
