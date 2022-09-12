import torch
import torch.nn as nn


_out_channel_list = [32,  64, 128, 256, 512]
_kernel_size_list = [3,    5,   7,   9,   9]
_stride_list      = [1,    1,   2,   3,   5]
_padding_list     = [1,    0,   0,   0,   0]

class ConvEncoder(nn.Module):
  def __init__(self, in_shape) -> None:
    super(ConvEncoder, self).__init__()
    
    in_channels, in_height, in_width = in_shape
    
    self.conv_layers = nn.ModuleList()
    curr_channels = in_channels
    curr_width    = in_width
    curr_height   = in_height

    self.output_sizes = []
    for out_channels, kernel_size, stride, padding in zip(_out_channel_list, _kernel_size_list, _stride_list, _padding_list):
      self.conv_layers.append(nn.Conv2d(
        in_channels=curr_channels, out_channels=out_channels, 
        kernel_size=kernel_size, stride=stride, padding=padding, bias=False
      ))
      self.conv_layers.append(nn.BatchNorm2d(out_channels))
      self.conv_layers.append(nn.ReLU(inplace=True))
      
      curr_width  = int((curr_width-kernel_size + 2*padding) / stride + 1)
      curr_height = int((curr_height-kernel_size + 2*padding) / stride + 1)
      curr_channels = out_channels
      self.output_sizes.append((curr_channels, curr_height, curr_width))

    self.conv_output_shape = (curr_channels, curr_height, curr_width)
    self.conv_output_size  = curr_width*curr_height*curr_channels
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.conv_layers:
      x = layer(x)
    return torch.flatten(x, 1)

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
  def __init__(self, encoder_input_shape, encoder_sizes) -> None:
    super(ConvDecoder, self).__init__()
    
    self.conv_layers = nn.ModuleList()
    
    for i in reversed(range(1,len(_padding_list))):
      in_channels = _out_channel_list[i]
      out_channels = _out_channel_list[i-1]
      kernel_size = _kernel_size_list[i]
      stride = _stride_list[i]
      padding = _padding_list[i]
      output_size = encoder_sizes[i-1]
      
      self.conv_layers.append(ConvT2dOutSize(nn.ConvTranspose2d(
        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
        stride=stride, padding=padding, output_padding=0
      ), (output_size[1],output_size[2])))
      self.conv_layers.append(nn.BatchNorm2d(out_channels))
      self.conv_layers.append(nn.ReLU(inplace=True))
      
    encoder_in_channels,_,_ = encoder_input_shape
    self.final_layer = nn.Sequential(
      nn.ConvTranspose2d(
        in_channels=_out_channel_list[0], out_channels=encoder_in_channels, kernel_size=_kernel_size_list[0],
        stride=_stride_list[0], padding=_padding_list[0], output_padding=0
      ),
      nn.Tanh()
    )
    
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    for layer in self.conv_layers:
      x = layer(x)
    return self.final_layer(x)
    

class DoomVAE(nn.Module):
  def __init__(self, input_shape, latent_dim=256) -> None:
    super(DoomVAE, self).__init__()
    
    self.latent_dim = latent_dim
    
    # Encoder
    self.conv_encoder = ConvEncoder(input_shape)
    
    # Latent representation 
    prelatent_size = self.conv_encoder.conv_output_size
    self.z_mu      = nn.Linear(prelatent_size, latent_dim)
    self.z_logvar   = nn.Linear(prelatent_size, latent_dim)
    
    # Decoder
    self.decoder_input = nn.Linear(latent_dim, prelatent_size)
    self.conv_decoder = ConvDecoder(input_shape, self.conv_encoder.output_sizes)
    
    
  def encode(self, input:torch.Tensor) -> list[torch.Tensor]:
    """
    Encodes the input by passing it through the encoder network,
    resulting in the latent representation for it.
    Args:
        input (torch.Tensor): Input tensor to the encoder [B x C x H x W]
    Returns:
        list[torch.Tensor]: List of latent representations [mu, std_dev]
    """
    result = self.conv_encoder(input)
    
    # Split the result of the flattened convolutional encoding into mu (mean) 
    # and sigma (standard deviation) of the latent Gaussian distribution
    mu = self.z_mu(result)
    logvar = self.z_logvar(result)
    return [mu, logvar]
  
  def decode(self, z:torch.Tensor) -> torch.Tensor:
    """
    Maps the given latent representation back into image space.
    Args:
        z (torch.Tensor): [B x D], if mu and sigma values are present then
        you must call 'reparameterize' in order to convert them to z.
    Returns:
        torch.Tensor: [B x C x H x W] tensor representing the image.
    """
    conv_ch, conv_h, conv_w = self.conv_encoder.conv_output_shape
    result = self.decoder_input(z)
    result = result.view(-1, conv_ch, conv_h, conv_w)
    return self.conv_decoder(result)
    
  def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization to sample from a Gaussian distribution N(mu,sigma) via a random sample in N(0,1).
    Args:
        mu (torch.Tensor): The mean of the latent Gaussian [B x D]
        sigma (torch.Tensor): The standard deviation of the latent Gaussian [B x D]
    Returns:
        torch.Tensor: 'z' latent representation as a sampled tensor [B x D]
    """
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    return epsilon * std + mu
  
  def forward(self, input:torch.Tensor) -> list[torch.Tensor]:
    mu, logvar = self.encode(input)
    z = self.reparameterize(mu, logvar)
    return [self.decode(z), input, mu, logvar]


  def loss_function(self, reconstruction, input, mu, logvar):
    reconst_loss = nn.functional.mse_loss(reconstruction, input)
    kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1), dim=0)
    return reconst_loss + kld_loss
  
  def sample(self, num_samples:int, device:torch.device) -> torch.Tensor:
    """
    Samples from the latent space and returns the corresponding image space map.
    Args:
        num_samples (int): The number of samples
        device (torch.device): The device running the model
    Returns:
        torch.Tensor: Resulting image space map [num_samples x C x H x W]
    """
    z = torch.randn(num_samples, self.latent_dim).to(device)
    return self.decode(z)
  
  def generate(self, x:torch.Tensor) -> torch.Tensor:
    """
    Given an input image, returns the reconstructed image.
    Args:
        x (torch.Tensor): The input image tensor [B x C x H x W]
    Returns:
        torch.Tensor: The reconstructed image tensor [B x C x H x W]
    """
    return self.forward(x)[0]