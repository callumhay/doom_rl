
import torch
import torch.nn as nn
import torch.nn.functional as F


class SDVAELoss(nn.Module):
  def __init__(self, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0):
    super().__init__()
    self.kl_weight = kl_weight
    self.pixel_weight = pixelloss_weight
    # output log variance
    self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
    
  def forward(self, inputs, reconstructions, posteriors):
    rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
    nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
    nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

    kl_loss = posteriors.kl()
    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

    return nll_loss + self.kl_weight * kl_loss, nll_loss, kl_loss
