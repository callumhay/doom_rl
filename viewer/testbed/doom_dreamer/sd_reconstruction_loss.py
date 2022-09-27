
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SDReconstructionLoss(nn.Module):
  def __init__(self):
    super().__init__()
    # output log variance
    self.logvar = nn.Parameter(torch.zeros(size=()))
    
  def reconstruction_nll_loss(self, inputs, reconstructions):
    rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
    nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
    seq_batch_size = np.prod(nll_loss.shape[:-3])
    nll_loss = torch.sum(nll_loss) / seq_batch_size
    return nll_loss


