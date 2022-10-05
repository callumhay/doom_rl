
import torch
import numpy as np

class DiagonalGaussianDistribution(object):
  def __init__(self, parameters, chunk_dim):
    #assert not torch.any(parameters.isnan())
    self.parameters = parameters
    self.mean, self.logvar = torch.chunk(parameters, 2, dim=chunk_dim)
    self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)

  def embedding(self, flatten_dim):
    return torch.flatten(torch.cat([self.mean, self.logvar], dim=-1), flatten_dim)

  def sample(self):
    x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
    return x

  def kl(self, dims=[2,3,4]):
    return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims)

  def nll(self, sample, dims=[2,3,4]):
    logtwopi = np.log(2.0 * np.pi)
    return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

  def mode(self):
    return self.mean
