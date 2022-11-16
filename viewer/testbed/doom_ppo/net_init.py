import torch.nn as nn
import numpy as np

def fc_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, bias_const)
  return layer

def fc_layer_init_rnd(layer, w_min_max=(-1,1), b_min_max=(-1,1)):
  nn.init.uniform_(layer.weight, w_min_max[0], w_min_max[1])
  nn.init.uniform_(layer.bias,   b_min_max[0], b_min_max[1])
  return layer

def fc_layer_init_norm(layer, w_std=1, b_std=0.1):
  nn.init.normal_(layer.weight, 0, w_std)
  nn.init.normal_(layer.bias, 0, b_std)
  return layer

def conv_layer_init(layer):
  nn.init.kaiming_uniform_(layer.weight, nonlinearity='linear') # Change back to kaiming_normal?
  nn.init.constant_(layer.bias, 0)
  return layer
