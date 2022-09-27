from typing import Dict, Union, Tuple, List
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
#import numpy as np

VAEDiscState = namedtuple('VAEDiscState', ['logit', 'stoch', 'deter'])
VAEContState = namedtuple('VAEContState', ['mean', 'std', 'stoch', 'deter'])  
VAEState = Union[VAEDiscState, VAEContState]

_DISCRETE_NET_TYPE   = 'discrete'
_CONTINUOUS_NET_TYPE = 'continuous'

# Config
_latent_space_size = 1048
_embedding_size = _latent_space_size
_network_type = _CONTINUOUS_NET_TYPE
_deter_size = 128
_stoch_size = 1048
_modelstate_size = _stoch_size + _deter_size
_temporal_hidden_size  = 1048 # node_size
_activation_fn = nn.ELU

_action_size   = 8 # TODO: Get this number from the DoomEnv
_category_size = 512
_class_size    = 512

class DoomVAE(nn.Module):
  def __init__(self, input_shape) -> None:
    super(DoomVAE, self).__init__()
    
    self.network_type = _network_type
    self.input_shape = input_shape
    self.latent_dim = _latent_space_size

    # Encoder
    self.conv_encoder = ConvEncoder(input_shape, _latent_space_size) 
    # Decoder
    self.conv_decoder = ConvDecoder(self.input_shape, self.conv_encoder.conv_output_shape, self.conv_encoder.output_sizes, _modelstate_size)
    
    # Temporal prior and posterior networks
    self.fc_prior = self._build_temporal_prior()
    self.fc_posterior = self._build_temporal_posterior()
    self.rnn = nn.GRUCell(_stoch_size, _deter_size)
    
    
  def _build_temporal_prior(self):
    """
    Take in latest deterministic state and output prior over stochastic state
    """
    temporal_prior = [nn.Linear(_deter_size, _temporal_hidden_size)]
    temporal_prior += [_activation_fn()]
    if self.network_type == _DISCRETE_NET_TYPE:
        temporal_prior += [nn.Linear(_temporal_hidden_size, _stoch_size)]
    elif self.network_type == _CONTINUOUS_NET_TYPE:
          temporal_prior += [nn.Linear(_temporal_hidden_size, 2 * _stoch_size)]
    return nn.Sequential(*temporal_prior)

  def _build_temporal_posterior(self):
    """
    Take in latest embedded observation and deterministic state and output posterior over stochastic states
    """
    temporal_posterior = [nn.Linear(_deter_size + _embedding_size, _temporal_hidden_size)]
    temporal_posterior += [_activation_fn()]
    if self.network_type == _DISCRETE_NET_TYPE:
        temporal_posterior += [nn.Linear(_temporal_hidden_size, _stoch_size)]
    elif self.network_type == _CONTINUOUS_NET_TYPE:
        temporal_posterior += [nn.Linear(_temporal_hidden_size, 2 * _stoch_size)]
    return nn.Sequential(*temporal_posterior)   
  
  def _build_embed_state_action(self):
    """
    Take in previous stochastic state and previous action and embed it to deterministic size for rnn input
    """
    return nn.Sequential(
      nn.Linear(_stoch_size + _action_size, _deter_size),
      _activation_fn()
    )


  # Interface methods for training and inference ******************************

  def generate(self, observation:torch.Tensor) -> torch.Tensor:
    """
    Given an input image, returns the reconstructed image.
    Args:
        observation (torch.Tensor): The input image tensor [B x C x H x W]
    Returns:
        torch.Tensor: The reconstructed image tensor [B x C x H x W]
    """
    return self.forward(observation, observation.device)[0]
      
  def forward(self, observation:torch.Tensor, device:torch.device, kl_loss_scale=1):
  
    batch_size = 1#len(observation)
    seq_len = len(observation) # TODO: CHANGE THIS.
    
    # TODO: CHANGE THIS. For now we unsqueeze the embed so that the shape reflects
    # (seq_len, batch_size=1, _embedded_size)
    embed = self.conv_encoder(observation).unsqueeze(1)
    
    prev_state = self._init_state(batch_size, device)
    prior, posterior = self.rollout_observation(seq_len, embed, prev_state)
    posterior_modelstate = self._get_model_state(posterior)
    
    # TODO: CHANGE THIS. We have to re-squeeze the model state to remove the sequence length
    # in order to pass it through the decoder
    temp = posterior_modelstate[:-1].view(-1,posterior_modelstate.shape[-1])
    observation_reconst = self.conv_decoder(temp)
    
    # NOTE: This is from the sigma-vae paper (https://orybkin.github.io/sigma-vae)
    # To go back to using the Dreamer v2 version: replace sigma with a value of 1, replace the kl_loss_scale with ~0.1
    sigma = (-6 + F.softplus(((observation[:-1]-observation_reconst)**2).mean([0,1,2,3], keepdim=True).sqrt().log()+6)).exp()
    observation_dist = td.Independent(td.Normal(observation_reconst, sigma), len(self.input_shape))
    obs_loss = -torch.mean(observation_dist.log_prob(observation[:-1]))
    kl_loss  = self._kl_loss(prior, posterior)
    
    return [observation_reconst, obs_loss + kl_loss_scale*kl_loss, obs_loss, kl_loss]
    

  def _kl_loss(self, prior, posterior):
    prior_dist     = self._get_distribution(prior)
    posterior_dist = self._get_distribution(posterior)
    alpha = 0.8
    kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(self._get_distribution(self._detach_state(posterior)), prior_dist))
    kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(posterior_dist, self._get_distribution(self._detach_state(prior))))
    kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs
    return kl_loss
  
  
  #def imagine(self, prev_action:torch.Tensor, prev_state:VAEState, non_terminal:torch.Tensor=True) -> VAEState:
  def imagine(self, prev_state:VAEState) -> VAEState:
    
    #prev_state_action_embed = self.fc_embed_state_action(torch.cat([prev_state.stoch*non_terminal, prev_action], dim=-1))
    #deter_state = self.rnn(prev_state_action_embed, prev_state.deter*non_terminal)
    deter_state = self.rnn(prev_state.stoch, prev_state.deter)
    
    if self.network_type == _DISCRETE_NET_TYPE:
      prior_logit = self.fc_prior(deter_state)
      stats = {'logit': prior_logit}
      prior_stoch_state = self._get_stochastic_state(stats)
      prior_state = VAEDiscState(prior_logit, prior_stoch_state, deter_state)
      
    elif self.network_type == _CONTINUOUS_NET_TYPE:
      prior_mean, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
      stats = {'mean': prior_mean, 'std': prior_std}
      prior_stoch_state, std = self._get_stochastic_state(stats)
      prior_state = VAEContState(prior_mean, std, prior_stoch_state, deter_state)
      
    return prior_state
  
  
  #def observe(self, observation_embed:torch.Tensor, prev_action:torch.Tensor, prev_non_terminal:torch.Tensor, prev_state:VAEState) -> tuple(VAEState, VAEState):
  def observe(self, observation_embed:torch.Tensor, prev_state:VAEState) -> Tuple[VAEState, VAEState]:
    prior_state = self.imagine(prev_state)
    deter_state = prior_state.deter
    # NOTE: dim=-1 means concatenation happens along the last dim
    x = torch.cat((deter_state, observation_embed), dim=-1)
    
    if self.network_type == _DISCRETE_NET_TYPE:
      posterior_logit = self.fc_posterior(x)
      stats = {'logit': posterior_logit}
      posterior_stoch_state = self._get_stochastic_state(stats)
      posterior_state = VAEDiscState(posterior_logit, posterior_stoch_state, deter_state)
      
    elif self.network_type == _CONTINUOUS_NET_TYPE:
      #_network_type == _CONTINUOUS_NET_TYPE
      posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1)
      stats = {'mean': posterior_mean, 'std': posterior_std}
      posterior_stoch_state, std = self._get_stochastic_state(stats)
      posterior_state = VAEContState(posterior_mean, std, posterior_stoch_state, deter_state)
    return prior_state, posterior_state
  
  
  def rollout_observation(self, seq_len:int, observation_embed:torch.Tensor, prev_state:VAEState) -> Tuple[VAEState, VAEState]:
    priors = []
    posteriors = []
    for t in range(seq_len):
      prior_state, posterior_state = self.observe(observation_embed[t], prev_state)
      priors.append(prior_state)
      posteriors.append(posterior_state)
      prev_state = posterior_state
    prior = self._stack_states(priors)
    posterior = self._stack_states(posteriors)
    return prior, posterior
  

  def rollout_imagination(self, horizon:int, actor:nn.Module, prev_state:VAEState):
    curr_state = prev_state
    next_states = []
    action_entropys = []
    imagination_log_probs = []
    for t in range(horizon):
      action, action_distribution = actor((self._get_model_state(curr_state)).detach())
      curr_state = self.imagine(action, curr_state)
      next_states.append(curr_state)
      action_entropys.append(action_distribution.entropy())
      imagination_log_probs.append(action_distribution.log_prob(torch.round(action.detach())))
    
    next_states = self._stack_states(next_states, dim=0)
    action_entropys = torch.stack(action_entropys, dim=0)
    imagination_log_probs = torch.stack(imagination_log_probs, dim=0)
    return next_states, imagination_log_probs, action_entropys


  def _get_distribution(self, state:VAEState) -> td.Independent:
    if self.network_type == _DISCRETE_NET_TYPE:
      shape = state.logit.shape
      logit = torch.reshape(state.logit, shape=(*shape[:-1], _category_size, _class_size))
      return td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1)
    elif self.network_type == _CONTINUOUS_NET_TYPE:
      return td.Independent(td.Normal(state.mean, state.std), 1)
  
  def _get_model_state(self, state:VAEState) -> VAEState:
    # Model is calculated the same for both discrete and continuous states
    return torch.cat((state.deter, state.stoch), dim=-1)
    
  def _get_stochastic_state(self, stats:Dict) -> torch.Tensor:
    if self.network_type == _DISCRETE_NET_TYPE:
       logit = stats['logit']
       shape = logit.shape
       logit = torch.reshape(logit, shape = (*shape[:-1], _category_size, _class_size))
       dist = td.OneHotCategorical(logits=logit)        
       stoch = dist.sample()
       stoch += dist.probs - dist.probs.detach()
       return torch.flatten(stoch, start_dim=-2, end_dim=-1)   
    elif self.network_type == _CONTINUOUS_NET_TYPE:
      mean = stats['mean']
      std  = stats['std']
      std = nn.functional.softplus(std) + 0.1
      return mean + std*torch.randn_like(mean), std
  
  def _stack_states(self, states:List[VAEState], dim=0) -> VAEState:
    if self.network_type == _DISCRETE_NET_TYPE:
      return VAEDiscState(
        torch.stack([state.logit for state in states], dim=dim),
        torch.stack([state.stoch for state in states], dim=dim),
        torch.stack([state.deter for state in states], dim=dim),
      )
    elif self.network_type == _CONTINUOUS_NET_TYPE: 
      return VAEContState(
        torch.stack([state.mean  for state in states], dim=dim),
        torch.stack([state.std   for state in states], dim=dim),
        torch.stack([state.stoch for state in states], dim=dim),
        torch.stack([state.deter for state in states], dim=dim),
      )

  def _init_state(self, batch_size, device) -> VAEState:
    if self.network_type == _DISCRETE_NET_TYPE:
      return VAEDiscState(
        torch.zeros(batch_size, _stoch_size).to(device),
        torch.zeros(batch_size, _stoch_size).to(device),
        torch.zeros(batch_size, _deter_size).to(device),
      )
    elif self.network_type == _CONTINUOUS_NET_TYPE:
      return VAEContState(
        torch.zeros(batch_size, _stoch_size).to(device),
        torch.zeros(batch_size, _stoch_size).to(device),
        torch.zeros(batch_size, _stoch_size).to(device),
        torch.zeros(batch_size, _deter_size).to(device),
      )
    
  def _detach_state(self, state:VAEState) -> VAEState:
    if self.network_type == _DISCRETE_NET_TYPE:
      return VAEDiscState(
        state.logit.detach(),
        state.stoch.detach(),
        state.deter.detach(),
      )
    elif self.network_type == _CONTINUOUS_NET_TYPE:
      return VAEContState(
        state.mean.detach(),
        state.std.detach(),
        state.stoch.detach(),
        state.deter.detach(),
      )
