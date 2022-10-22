import torch.nn as nn
import numpy as np

def fc_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, bias_const)
  return layer

def conv_layer_init(layer):
  nn.init.kaiming_normal_(layer.weight, nonlinearity='linear')
  nn.init.constant_(layer.bias, 0)
  return layer
