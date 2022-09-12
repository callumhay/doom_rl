import torch
import torch.nn as nn

class ConvNetwork(nn.Module):
  def __init__(self, in_shape) -> None:
    super().__init__()
    
    in_channels, in_height, in_width = in_shape
    
    out_channel_list = [32, 64, 128, 256, 256, 512, 512]
    kernel_size_list = [3,   5,   5,   7,   7,   7,   9]
    stride_list      = [1,   1,   2,   1,   2,   3,   1]
    padding_list     = [1,   1,   1,   1,   1,   1,   1]
    
    self.layers = nn.ModuleList()
    curr_channels = in_channels
    curr_width = in_width
    curr_height = in_height

    for out_channels, kernel_size, stride, padding in zip(out_channel_list, kernel_size_list, stride_list, padding_list):
      self.layers.append(nn.Conv2d(
        in_channels=curr_channels, out_channels=out_channels, 
        kernel_size=kernel_size, stride=stride, padding=padding, bias=False
      ))
      self.layers.append(nn.BatchNorm2d(out_channels))
      self.layers.append(nn.ReLU(inplace=True))
      
      curr_width  = int((curr_width-kernel_size + 2*padding) / stride + 1)
      curr_height = int((curr_height-kernel_size + 2*padding) / stride + 1)
      curr_channels = out_channels

    self.output_size = curr_width*curr_height*curr_channels
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      x = layer(x)
    return torch.flatten(x, 1)

class DoomClassifier(nn.Module):
  def __init__(self, input_shape, num_classes=256, hidden_sizes=[1024]):
    super().__init__()

    self.conv_layer = ConvNetwork(input_shape)
    self.hidden_layers = nn.ModuleList([
      nn.Dropout(0.2),
      nn.ReLU(inplace=True)
    ])
    last_hidden_size = self.conv_layer.output_size
    for hidden_size in hidden_sizes:
      self.hidden_layers.append(nn.Linear(last_hidden_size, hidden_size))
      self.hidden_layers.append(nn.Dropout(0.2))
      self.hidden_layers.append(nn.ReLU(inplace=True))
      last_hidden_size = hidden_size
    
    self.linear_out = torch.nn.Linear(last_hidden_size, num_classes)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv_layer(x)
    for layer in self.hidden_layers:
      x = layer(x)
    return self.linear_out(x)


