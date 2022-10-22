import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, bias_const)
  return layer

def _normalize(in_channels, num_groups=16):
  return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def _nonlinearity(x):
  return x*torch.sigmoid(x)

class Upsample(nn.Module):
  def __init__(self, in_channels, with_conv):
    super().__init__()
    self.with_conv = with_conv
    if self.with_conv:
      self.conv = _layer_init(torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))

  def forward(self, x):
    x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
    if self.with_conv:
      x = self.conv(x)
    return x


class Downsample(nn.Module):
  def __init__(self, in_channels, with_conv):
    super().__init__()
    self.with_conv = with_conv
    if self.with_conv:
      self.conv = _layer_init(torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0))

  def forward(self, x):
    if self.with_conv:
      # no asymmetric padding in torch conv, must do it ourselves
      pad = (0,1,0,1)
      x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
      x = self.conv(x)
    else:
      x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
    return x

class ResnetBlock(nn.Module):
  def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
    super().__init__()
    self.in_channels = in_channels
    out_channels = in_channels if out_channels is None else out_channels
    self.out_channels = out_channels
    self.use_conv_shortcut = conv_shortcut

    self.norm1 = _normalize(in_channels)
    self.conv1 = _layer_init(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
    
    if temb_channels > 0: self.temb_proj = _layer_init(torch.nn.Linear(temb_channels, out_channels))
    
    self.norm2 = _normalize(out_channels)
    self.dropout = torch.nn.Dropout(dropout)
    self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    if self.in_channels != self.out_channels:
      if self.use_conv_shortcut:
        self.conv_shortcut = _layer_init(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
      else:
        self.nin_shortcut = _layer_init(torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0))

  def forward(self, x, temb):
    h = x
    h = self.norm1(h)
    h = _nonlinearity(h)
    h = self.conv1(h)

    if temb is not None:
        h = h + self.temb_proj(_nonlinearity(temb))[:,:,None,None]

    h = self.norm2(h)
    h = _nonlinearity(h)
    h = self.dropout(h)
    h = self.conv2(h)

    if self.in_channels != self.out_channels:
      if self.use_conv_shortcut:
        x = self.conv_shortcut(x)
      else:
        x = self.nin_shortcut(x)

    return x+h

class AttnBlock(nn.Module):
  def __init__(self, in_channels):
    super().__init__()
    self.in_channels = in_channels

    self.norm = _normalize(in_channels)
    self.q = _layer_init(torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0))
    self.k = _layer_init(torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0))
    self.v = _layer_init(torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0))
    self.proj_out = _layer_init(torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0))


  def forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    # compute attention
    b,c,h,w = q.shape
    q = q.reshape(b,c,h*w)
    q = q.permute(0,2,1)   # b,hw,c
    k = k.reshape(b,c,h*w) # b,c,hw
    w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
    w_ = w_ * (int(c)**(-0.5))
    w_ = torch.nn.functional.softmax(w_ - w_.amax(keepdims=True), dim=2)

    # attend to values
    v = v.reshape(b,c,h*w)
    w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
    h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
    h_ = h_.reshape(b,c,h,w)

    h_ = self.proj_out(h_)

    return x+h_
  
  
class SDEncoder(nn.Module):
  def __init__(self, env, args):
    
    super().__init__()
    
    in_channels, res_h, res_w = env.observation_shape()
    starting_channels = args.starting_channels
    num_res_blocks = args.num_res_blocks
    ch_mult = args.ch_mult
    dropout = args.dropout
    z_channels = args.z_channels

    self.ch = starting_channels
    self.temb_ch = 0
    self.num_resolutions = len(ch_mult)
    self.num_res_blocks = num_res_blocks
    self.in_channels = in_channels

    # downsampling
    self.conv_in = _layer_init(torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1))

    curr_res = np.array([res_h, res_w])
    in_ch_mult = (1,)+tuple(ch_mult)
    self.in_ch_mult = in_ch_mult
    self.down = nn.ModuleList()
    for i_level in range(self.num_resolutions):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_in = self.ch*in_ch_mult[i_level]
      block_out = self.ch*ch_mult[i_level]
      for _ in range(self.num_res_blocks):
        block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
        block_in = block_out

      down = nn.Module()
      down.block = block
      down.attn = attn
      if i_level != self.num_resolutions-1:
        down.downsample = Downsample(block_in, True)
        curr_res = curr_res // 2
      self.down.append(down)

    # middle
    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
    self.mid.attn_1  = AttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

    # end
    self.norm_out = _normalize(block_in)
    self.conv_out = _layer_init(torch.nn.Conv2d(block_in, 2*z_channels, kernel_size=3, stride=1, padding=1))

  def forward(self, x):
    # timestep embedding
    temb = None

    # downsampling
    hs = [self.conv_in(x)]
    for i_level in range(self.num_resolutions):
      for i_block in range(self.num_res_blocks):
        h = self.down[i_level].block[i_block](hs[-1], temb)
        if len(self.down[i_level].attn) > 0:
          h = self.down[i_level].attn[i_block](h)
        hs.append(h)
      if i_level != self.num_resolutions-1:
        hs.append(self.down[i_level].downsample(hs[-1]))

    # middle
    h = hs[-1]
    h = self.mid.block_1(h, temb)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h, temb)

    # end
    h = self.norm_out(h)
    h = _nonlinearity(h)
    h = self.conv_out(h)
    return h
  

class SDDecoder(nn.Module):
  def __init__(self, env, args): 
    super().__init__()
    
    out_channels, res_h, res_w = env.observation_shape()
    ch_mult = args.ch_mult
    dropout = args.dropout
    z_channels = args.z_channels

    self.ch = args.starting_channels
    self.temb_ch = 0
    self.num_resolutions = len(args.ch_mult)
    self.num_res_blocks = args.num_res_blocks
    self.out_channels = out_channels

    # compute in_ch_mult, block_in and curr_res at lowest res
    block_in = self.ch*ch_mult[self.num_resolutions-1]
    curr_res = np.array([res_h, res_w]) // 2**(self.num_resolutions-1)
    self.z_shape = (1, z_channels, curr_res[0], curr_res[1])
    print(f"Working with z of shape {self.z_shape} = {np.prod(self.z_shape)} dimensions.")

    # z to block_in
    self.conv_in = _layer_init(torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1))

    # middle
    self.mid = nn.Module()
    self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)
    self.mid.attn_1  = AttnBlock(block_in)
    self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout)

    # upsampling
    self.up = nn.ModuleList()
    for i_level in reversed(range(self.num_resolutions)):
      block = nn.ModuleList()
      attn = nn.ModuleList()
      block_out = self.ch*ch_mult[i_level]
      for _ in range(self.num_res_blocks+1):
          block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, temb_channels=self.temb_ch, dropout=dropout))
          block_in = block_out
      up = nn.Module()
      up.block = block
      up.attn = attn
      if i_level != 0:
        up.upsample = Upsample(block_in, True)
        curr_res = curr_res * 2
      self.up.insert(0, up) # prepend to get consistent order

    # end
    self.norm_out = _normalize(block_in)
    self.conv_out = _layer_init(torch.nn.Conv2d(block_in, self.out_channels, kernel_size=3, stride=1, padding=1))

  def forward(self, z):
    # timestep embedding
    temb = None

    # z to block_in
    h = self.conv_in(z)

    # middle
    h = self.mid.block_1(h, temb)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h, temb)

    # upsampling
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks+1):
        h = self.up[i_level].block[i_block](h, temb)
        if len(self.up[i_level].attn) > 0:
          h = self.up[i_level].attn[i_block](h)
      if i_level != 0:
        h = self.up[i_level].upsample(h)

    h = self.norm_out(h)
    h = _nonlinearity(h)
    h = self.conv_out(h)

    return h


class SDVAE(nn.Module):
  def __init__(self, env, args) -> None:
    super().__init__()
    
    z_channels = args.z_channels
    self.encoder = SDEncoder(env, args)
    self.decoder = SDDecoder(env, args)
    self.loss = SDVAELoss(args)
    
    self.is_categorical = args.categorical
    if self.is_categorical:
      self.N = args.num_categories
      self.K = args.num_classes
      self.embed_dim = self.N*self.K
      self.embed_shape = (self.embed_dim,) + self.decoder.z_shape[-2:]
      in_z_shape = (int(self.decoder.z_shape[1]),) + self.decoder.z_shape[-2:]
      linear_in = np.prod(in_z_shape, dtype=np.int32)
      self.quant_conv = nn.Sequential(
        nn.Flatten(),
        _layer_init(nn.Linear(2*linear_in, self.embed_dim)),
      )
      self.post_quant_conv = nn.Sequential(
        _layer_init(nn.Linear(self.embed_dim, linear_in)),
        nn.Unflatten(dim=-1, unflattened_size=(int(in_z_shape[0]), int(in_z_shape[1]), int(in_z_shape[2]))),
      )
    else:
      self.embed_dim = args.embed_dim
      self.quant_conv = torch.nn.Conv2d(2*z_channels, 2*self.embed_dim, 1)
      self.post_quant_conv = torch.nn.Conv2d(self.embed_dim , z_channels, 1)
      
  def encode(self, x):
    h = self.encoder(x)
    h = self.quant_conv(h)
    if self.is_categorical:
      logits = h.view(-1, self.N, self.K)
      posterior = GumbelCategoricalDistribution(logits)
    else:
      posterior = DiagonalGaussianDistribution(h)
    return posterior
  
  def decode(self, z):
    if self.is_categorical:
      z = z.view(-1, self.embed_dim)
    z = self.post_quant_conv(z)
    dec = self.decoder(z)
    return dec
  
  def forward(self, input, sample_posterior=True):
    posterior = self.encode(input)
    if sample_posterior:
      z = posterior.sample()
    else:
      z = posterior.mode()
    dec = self.decode(z)
    return dec, posterior

  def training_step(self, inputs):
    reconstructions, posterior = self(inputs)
    loss = self.loss(inputs, reconstructions, posterior)
    
    # The following works very poorly compared to self.loss:
    #reconst_loss = torch.mean(nn.functional.mse_loss(reconstructions, inputs, reduction='none'), dim=[1,2,3])
    #kl_loss = -0.5 * torch.sum(1 + posterior.logvar - posterior.mean**2 - posterior.var, dim=[1,2,3])
    #total_loss = (reconst_loss + 0.00025 * kl_loss).mean(dim=0)
    #loss = (total_loss, reconst_loss.mean(dim=0), kl_loss.mean(dim=0))
    
    return *loss, reconstructions, posterior


class SDVAELoss(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.kl_weight = args.kl_weight
    self.logvar = nn.Parameter(torch.zeros(size=()))
    
  def forward(self, inputs, reconstructions, posteriors):
    rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
    nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
    #nll_loss = nll_loss + torch.clamp(self.logvar, min=-nll_loss) # This doesn't work.
    nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
    kl_loss = posteriors.kl()
    kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
    return nll_loss + self.kl_weight * kl_loss, nll_loss, kl_loss
  

class GumbelCategoricalDistribution(object):
  def __init__(self, logits: torch.Tensor, temperature: float=1.0) -> None:
    # b is batch, n is number of categorical distributions, k is number of classes
    self.b, self.n, self.k = logits.shape
    self.shape = (self.b*self.n, self.k)
    self.logits = logits.view(self.shape)
    self.temperature = temperature
    
  def gumbel_distribution_sample(self, eps=1e-20):
    U = torch.rand(self.shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)
  
  def sample(self):
    y = self.logits + self.gumbel_distribution_sample()
    y = torch.nn.functional.softmax(y / self.temperature, dim=-1)
    return y.view(self.b, self.n, self.k)
  
  def mode(self):
    y = self.logits
    y = torch.nn.functional.softmax(y / self.temperature, dim=-1)
    return y.view(self.b, self.n, self.k)
  
  def kl(self):
    q = td.Categorical(logits=self.logits)
    p = td.Categorical(probs=torch.full(self.shape, 1.0/self.k).cuda()) # uniform bunch of K-class categorical distributions
    kl = td.kl.kl_divergence(q, p) # kl is of shape [b*n]
    return kl.view(self.b, self.n)
  

class DiagonalGaussianDistribution(object):
  def __init__(self, parameters):
    self.parameters = parameters
    self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
    self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
    self.std = torch.exp(0.5 * self.logvar)
    self.var = torch.exp(self.logvar)

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
