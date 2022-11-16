import torch
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

def fc_layer_init_norm(layer, std=np.sqrt(2)):
  nn.init.normal_(layer.weight, 0, std)
  nn.init.normal_(layer.bias, 0, std)
  with torch.no_grad():
    layer.bias *= 0.01
  return layer

def fc_layer_init_xavier(layer, nonlinearity, gain_arg=None):
  nn.init.xavier_normal_(layer.weight, nn.init.calculate_gain(nonlinearity, gain_arg))
  nn.init.normal_(layer.bias, 0, 1)
  with torch.no_grad(): layer.bias *= 0.01
  return layer

def conv_layer_init(layer, nonlinearity='linear', a=0, bias_const=True):
  nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity, a=a) # Change back to kaiming_normal?
  if bias_const:
    nn.init.constant_(layer.bias, 0)
  else:
    nn.init.normal_(layer.bias, 0, 1)
    with torch.no_grad(): layer.bias *= 0.01
  return layer


def simple_conv_net(in_shape, out_size):
  out_channel_list = [32, 64, 64]
  kernel_size_list = [ 8,  4,  3]
  stride_list      = [ 4,  2,  1]
  padding_list     = [ 0,  0,  0]
  
  LEAKY_RELU_SLOPE = 0.25
  conv_net = nn.Sequential()
  curr_channels, curr_height, curr_width = in_shape
  for out_channels, kernel_size, stride, padding in zip(out_channel_list, kernel_size_list, stride_list, padding_list):
    # NOTE: Square vs. Rectangular kernels appear to have no noticable effect
    # If anything, rectangular is worse. Use square kernels for simplicity.
    kernel_size_h = kernel_size
    kernel_size_w = kernel_size
    conv_net.append(conv_layer_init(
      nn.Conv2d(curr_channels, out_channels, (kernel_size_h, kernel_size_w), stride),
      'leaky_relu', LEAKY_RELU_SLOPE, bias_const=False
    ))
    conv_net.append(nn.LeakyReLU(LEAKY_RELU_SLOPE, inplace=True))
    
    curr_width  = int((curr_width-kernel_size_w + 2*padding) / stride + 1)
    curr_height = int((curr_height-kernel_size_h + 2*padding) / stride + 1)
    curr_channels = out_channels
      
  conv_net.append(nn.Flatten())
  conv_output_size = curr_width*curr_height*curr_channels
  
  if out_size != None:
    conv_net.append(fc_layer_init(nn.Linear(conv_output_size, out_size)))
  else:
    out_size = conv_output_size
  return conv_net, out_size