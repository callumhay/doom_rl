
import torch
import numpy as np

class DiagonalGaussianDistribution(object):
  def __init__(self, parameters, chunk_dim):
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
