import torch
import torch.nn as nn
import numpy as np

from sd_encode_decode import SDEncoder, SDDecoder
from sd_loss import SDVAELoss


class SDVAE(nn.Module):
  def __init__(self, vae_config, embed_dim, ckpt_path=None, ignore_keys=[]) -> None:
    super().__init__()

    self.encoder = SDEncoder(**vae_config)
    self.decoder = SDDecoder(**vae_config)
    loss_config = vae_config['loss_config'] if 'loss_config' in vae_config else {}
    self.loss = SDVAELoss(**loss_config)
    self.quant_conv = torch.nn.Conv2d(2*vae_config["z_channels"], 2*embed_dim, 1)
    self.post_quant_conv = torch.nn.Conv2d(embed_dim, vae_config["z_channels"], 1)
    self.embed_dim = embed_dim
      
  def init_from_ckpt(self, path, ignore_keys=list()):
    sd = torch.load(path, map_location="cpu")["state_dict"]
    keys = list(sd.keys())
    for k in keys:
      for ik in ignore_keys:
        if k.startswith(ik):
          print("Deleting key {} from state_dict.".format(k))
          del sd[k]
    self.load_state_dict(sd, strict=False)
    print(f"Restored from {path}")

  def encode(self, x):
    h = self.encoder(x)
    moments = self.quant_conv(h)
    posterior = DiagonalGaussianDistribution(moments)
    return posterior
  
  def decode(self, z):
    z = self.post_quant_conv(z)
    dec = self.decoder(z)
    return dec
  
  def forward(self, input, sample_posterior=True):
    posterior = self.encode(input)
    if sample_posterior:
      z = posterior.sample()
    else:
      z = posterior.mode()
    dec = self.decode(z)
    return dec, posterior

  def training_step(self, inputs):
    reconstructions, posterior = self(inputs)
    return self.loss(inputs, reconstructions, posterior)


class DiagonalGaussianDistribution(object):
  def __init__(self, parameters):
    self.parameters = parameters
    self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
    self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)

  def sample(self):
    x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
    return x

  def kl(self, other=None):
    if other is None:
      return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
    else:
      return 0.5 * torch.sum(torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=[1, 2, 3])

  def nll(self, sample, dims=[1,2,3]):
    logtwopi = np.log(2.0 * np.pi)
    return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

  def mode(self):
    return self.mean
