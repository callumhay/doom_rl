import torch
import torch.nn as nn
import torch.distributions as td
import torch.nn.functional as F
import numpy as np

_out_channel_list = [32,  64, 128, 256, 512, 1024]
_kernel_size_list = [3,    3,   3,   3,   3,    3]
_stride_list      = [2,    2,   2,   2,   2,    2]
_padding_list     = [1,    1,   1,   1,   1,    1]

_use_batch_norm = False

class ConvEncoder(nn.Module):
  def __init__(self, in_shape, embedded_size) -> None:
    super(ConvEncoder, self).__init__()
    
    in_channels, in_height, in_width = in_shape
    
    self.encoder = nn.Sequential()
    curr_channels = in_channels
    curr_width    = in_width
    curr_height   = in_height

    self.output_sizes = []
    for out_channels, kernel_size, stride, padding in zip(_out_channel_list, _kernel_size_list, _stride_list, _padding_list):
      self.encoder.append(nn.Conv2d(
        in_channels=curr_channels, out_channels=out_channels, 
        kernel_size=kernel_size, stride=stride, padding=padding, bias=not _use_batch_norm
      ))
      if _use_batch_norm: self.encoder.append(nn.BatchNorm2d(out_channels))
      self.encoder.append(nn.ELU())
      
      curr_width  = int((curr_width-kernel_size + 2*padding) / stride + 1)
      curr_height = int((curr_height-kernel_size + 2*padding) / stride + 1)
      curr_channels = out_channels
      self.output_sizes.append((curr_channels, curr_height, curr_width))

    self.conv_output_shape = (curr_channels, curr_height, curr_width)
    self.conv_output_size  = curr_width*curr_height*curr_channels
    
    # Last layer converts the Convolution output into the embedded size
    self.fc_embedded = nn.Linear(self.conv_output_size, embedded_size)
    
    #for m in self.modules():
    #  if isinstance(m, nn.Conv2d):
    #    nn.init.uniform_(m.weight)

    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    return self.fc_embedded(torch.flatten(x, 1))

# Wrapper for pytorch ConvTranspose2d so that we can provide the specific
# output size to avoid ambiguous shapes during decoding
class ConvT2dOutSize(nn.Module):
  def __init__(self, conv, output_size) -> None:
    super(ConvT2dOutSize, self).__init__()
    self.output_size = output_size
    self.conv = conv
    
  def forward(self, x):
    return self.conv(x, output_size=self.output_size)
    

class ConvDecoder(nn.Module):
  def __init__(self, encoder_input_shape, encoder_output_shape, encoder_sizes, modelstate_size) -> None:
    super(ConvDecoder, self).__init__()
    
    prelatent_size = np.prod(encoder_output_shape)
    self.input_shape = encoder_output_shape
    self.fc_modelstate = nn.Linear(modelstate_size, prelatent_size)
    
    self.decoder = nn.Sequential()
    for i in reversed(range(1,len(_padding_list))):
      in_channels = _out_channel_list[i]
      out_channels = _out_channel_list[i-1]
      kernel_size = _kernel_size_list[i]
      stride = _stride_list[i]
      padding = _padding_list[i]
      output_size = encoder_sizes[i-1]
      
      self.decoder.append(ConvT2dOutSize(nn.ConvTranspose2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
        stride=stride, padding=padding, output_padding=1, bias=not _use_batch_norm
      ), (output_size[1],output_size[2])))
      if _use_batch_norm: self.decoder.append(nn.BatchNorm2d(out_channels))
      self.decoder.append(nn.ELU())
      
    encoder_in_channels,encoder_in_h,encoder_in_w = encoder_input_shape
    self.decoder.append(
      ConvT2dOutSize(nn.ConvTranspose2d(
        in_channels=_out_channel_list[0], out_channels=encoder_in_channels, kernel_size=_kernel_size_list[0],
        stride=_stride_list[0], padding=_padding_list[0]
      ),(encoder_in_h, encoder_in_w)),
    )
    # Applying a non-linearity here removes artifacts but does not result in as-good a representation
    #self.decoder.append(nn.Tanh()) # TODO: Get rid of this if it isn't working.
    
    #for m in self.modules():
    #  if isinstance(m, nn.Conv2d):
    #    nn.init.uniform_(m.weight)
    
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    conv_ch, conv_h, conv_w = self.input_shape
    x = self.fc_modelstate(x)
    x = x.view(-1, conv_ch, conv_h, conv_w)
    return self.decoder(x)

  def get_distribution(self, observations: torch.Tensor, reconstructions: torch.Tensor) -> td.Independent:
    sigma = (-6 + F.softplus(((observations-reconstructions)**2).mean([0,1,2,3], keepdim=True).sqrt().log()+6)).exp()
    return td.Independent(td.Normal(reconstructions, sigma), len(self.input_shape))
  