
from typing import Dict, Union, Tuple, List
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from config import CONTINUOUS_NET_TYPE, DISCRETE_NET_TYPE

RSSMDiscreteState = namedtuple('RSSMDiscreteState', ['logit', 'stoch', 'deter'])
RSSMContinuousState = namedtuple('RSSMContinuousState', ['mean', 'std', 'stoch', 'deter'])  
RSSMState = Union[RSSMDiscreteState, RSSMContinuousState]

class DoomRSSM(nn.Module):
  def __init__(self, action_size, embedding_size, device, info) -> None:
    super().__init__()
    
    self.device = device
    self.action_size = action_size
    self.embedding_size = embedding_size
    self.network_type = info['type']
    if self.network_type == CONTINUOUS_NET_TYPE:
        self.deter_size = info['deter_size']
        self.stoch_size = info['stoch_size']
        self.min_std = info['min_std']
    elif self.network_type == DISCRETE_NET_TYPE:
        self.deter_size = info['deter_size']
        self.class_size = info['class_size']
        self.category_size = info['category_size']
        self.stoch_size  = self.class_size*self.category_size
    else:
      raise NotImplementedError(f"Invalid RSSM network type '{self.network_type}")

    self.rnn = nn.GRUCell(self.deter_size, self.deter_size)
    self.fc_embed_state_action = self._build_embed_state_action(info)
    self.fc_prior = self._build_temporal_prior(info)
    self.fc_posterior = self._build_temporal_posterior(info)
    
  def _build_temporal_prior(self, info):
    """
    Take in latest deterministic state and output prior over stochastic state
    """
    hidden_size = info['hidden_size']
    activation_fn = info['activation_fn']
    
    temporal_prior = [nn.Linear(self.deter_size, hidden_size)]
    temporal_prior += [activation_fn()]
    if self.network_type == DISCRETE_NET_TYPE:
        temporal_prior += [nn.Linear(hidden_size, self.stoch_size)]
    elif self.network_type == CONTINUOUS_NET_TYPE:
          temporal_prior += [nn.Linear(hidden_size, 2 * self.stoch_size)]
    #temporal_prior += [activation_fn()]
    return nn.Sequential(*temporal_prior)

  def _build_temporal_posterior(self, info):
    """
    Take in latest embedded observation and deterministic state and output posterior over stochastic states
    """
    hidden_size = info['hidden_size']
    activation_fn = info['activation_fn']
    
    temporal_posterior = [nn.Linear(self.deter_size + self.embedding_size, hidden_size)]
    temporal_posterior += [activation_fn()]
    if self.network_type == DISCRETE_NET_TYPE:
        temporal_posterior += [nn.Linear(hidden_size, self.stoch_size)]
    elif self.network_type == CONTINUOUS_NET_TYPE:
        temporal_posterior += [nn.Linear(hidden_size, 2 * self.stoch_size)]
    #temporal_posterior += [activation_fn()]
    return nn.Sequential(*temporal_posterior)   
    
  def _build_embed_state_action(self, info):
    """
    Take in previous stochastic state and previous action and embed it to deterministic size for rnn input
    """
    activation_fn = info['activation_fn']
    return nn.Sequential(
      nn.Linear(self.stoch_size + self.action_size, self.deter_size),
      activation_fn()
    )
    
  
  def imagine(
    self, prev_action:torch.Tensor, prev_state:RSSMState, non_terminal:torch.Tensor=True
  ) -> RSSMState:
    prev_state_action_embed = self.fc_embed_state_action(torch.cat([prev_state.stoch*non_terminal, prev_action], dim=-1))
    deter_state = self.rnn(prev_state_action_embed, prev_state.deter*non_terminal)

    if self.network_type == DISCRETE_NET_TYPE:
      prior_logit = self.fc_prior(deter_state)
      stats = {'logit': prior_logit}
      prior_stoch_state = self._get_stochastic_state(stats)
      prior_state = RSSMDiscreteState(prior_logit, prior_stoch_state, deter_state)
      
    elif self.network_type == CONTINUOUS_NET_TYPE:
      prior_mean, prior_std = torch.chunk(self.fc_prior(deter_state), 2, dim=-1)
      stats = {'mean': prior_mean, 'std': prior_std}
      prior_stoch_state, std = self._get_stochastic_state(stats)
      prior_state = RSSMContinuousState(prior_mean, std, prior_stoch_state, deter_state)
      
    return prior_state
  
  
  def observe(
    self, observation_embed:torch.Tensor, prev_action:torch.Tensor,  
    prev_non_terminal:torch.Tensor, prev_state:RSSMState
  ) -> Tuple[RSSMState, RSSMState]:
    
    prior_state = self.imagine(prev_action, prev_state, prev_non_terminal)
    deter_state = prior_state.deter
    x = torch.cat((deter_state, observation_embed), dim=-1) # NOTE: dim=-1 means concatenation happens along the last dim
    
    if self.network_type == DISCRETE_NET_TYPE:
      posterior_logit = self.fc_posterior(x)
      stats = {'logit': posterior_logit}
      posterior_stoch_state = self._get_stochastic_state(stats)
      posterior_state = RSSMDiscreteState(posterior_logit, posterior_stoch_state, deter_state)

    elif self.network_type == CONTINUOUS_NET_TYPE:
      posterior_mean, posterior_std = torch.chunk(self.fc_posterior(x), 2, dim=-1)
      stats = {'mean': posterior_mean, 'std': posterior_std}
      posterior_stoch_state, std = self._get_stochastic_state(stats)
      posterior_state = RSSMContinuousState(posterior_mean, std, posterior_stoch_state, deter_state)
    return prior_state, posterior_state
  
  
  def rollout_observation(
    self, seq_len: int, observation_embed: torch.Tensor, 
    action: torch.Tensor, nonterminals: torch.Tensor, prev_state:RSSMState
  ) -> Tuple[RSSMState, RSSMState]:
    priors = []
    posteriors = []
    for t in range(seq_len):
      prev_action = action[t]*nonterminals[t]
      prior_state, posterior_state = self.observe(
        observation_embed[t], prev_action, nonterminals[t], prev_state
      )
      priors.append(prior_state)
      posteriors.append(posterior_state)
      prev_state = posterior_state
    prior = self.stack_states(priors, dim=0)
    posterior = self.stack_states(posteriors, dim=0)
    return prior, posterior
  

  def rollout_imagination(self, horizon:int, actor:nn.Module, prev_state:RSSMState):
    curr_state = prev_state
    next_states = []
    action_entropys = []
    imagination_log_probs = []
    for _ in range(horizon):
      temp_state = self.get_model_state(curr_state).detach()
      action, action_distribution = actor(temp_state)
      curr_state = self.imagine(action, curr_state)
      next_states.append(curr_state)
      action_entropys.append(action_distribution.entropy())
      imagination_log_probs.append(action_distribution.log_prob(torch.round(action.detach())))
    
    next_states = self.stack_states(next_states, dim=0)
    action_entropys = torch.stack(action_entropys, dim=0)
    imagination_log_probs = torch.stack(imagination_log_probs, dim=0)
    return next_states, imagination_log_probs, action_entropys

  def init_state(self, batch_size) -> RSSMState:
    if self.network_type == DISCRETE_NET_TYPE:
      return RSSMDiscreteState(
        torch.zeros(batch_size, self.stoch_size).to(self.device),
        torch.zeros(batch_size, self.stoch_size).to(self.device),
        torch.zeros(batch_size, self.deter_size).to(self.device),
      )
    elif self.network_type == CONTINUOUS_NET_TYPE:
      return RSSMContinuousState(
        torch.zeros(batch_size, self.stoch_size).to(self.device),
        torch.zeros(batch_size, self.stoch_size).to(self.device),
        torch.zeros(batch_size, self.stoch_size).to(self.device),
        torch.zeros(batch_size, self.deter_size).to(self.device),
      )
  
  def get_model_state(self, state:RSSMState) -> RSSMState:
    # Model is calculated the same for both discrete and continuous states
    return torch.cat((state.deter, state.stoch), dim=-1)
  
  
  def get_distribution(self, state:RSSMState) -> td.Independent:
    if self.network_type == DISCRETE_NET_TYPE:
      shape = state.logit.shape
      logit = torch.reshape(state.logit, shape=(*shape[:-1], self.category_size, self.class_size))
      return td.Independent(td.OneHotCategoricalStraightThrough(logits=logit), 1)
    elif self.network_type == CONTINUOUS_NET_TYPE:
      return td.Independent(td.Normal(state.mean, state.std), 1)
  
  def _get_stochastic_state(self, stats:Dict) -> torch.Tensor:
    if self.network_type == DISCRETE_NET_TYPE:
       logit = stats['logit']
       shape = logit.shape
       logit = torch.reshape(logit, shape = (*shape[:-1], self.category_size, self.class_size))
       dist = td.OneHotCategorical(logits=logit)        
       stoch = dist.sample()
       stoch += dist.probs - dist.probs.detach()
       return torch.flatten(stoch, start_dim=-2, end_dim=-1)   
    elif self.network_type == CONTINUOUS_NET_TYPE:
      mean = stats['mean']
      std  = stats['std']
      std = nn.functional.softplus(std) + 0.1
      return mean + std*torch.randn_like(mean), std
  
  def stack_states(self, states:List[RSSMState], dim=0) -> RSSMState:
    if self.network_type == DISCRETE_NET_TYPE:
      return RSSMDiscreteState(
        torch.stack([state.logit for state in states], dim=dim),
        torch.stack([state.stoch for state in states], dim=dim),
        torch.stack([state.deter for state in states], dim=dim),
      )
    elif self.network_type == CONTINUOUS_NET_TYPE: 
      return RSSMContinuousState(
        torch.stack([state.mean  for state in states], dim=dim),
        torch.stack([state.std   for state in states], dim=dim),
        torch.stack([state.stoch for state in states], dim=dim),
        torch.stack([state.deter for state in states], dim=dim),
      )

  def detach_state(self, state:RSSMState) -> RSSMState:
    if self.network_type == DISCRETE_NET_TYPE:
      return RSSMDiscreteState(
        state.logit.detach(),
        state.stoch.detach(),
        state.deter.detach(),
      )
    elif self.network_type == CONTINUOUS_NET_TYPE:
      return RSSMContinuousState(
        state.mean.detach(),
        state.std.detach(),
        state.stoch.detach(),
        state.deter.detach(),
      )
      
  def state_seq_to_batch(self, rssm_state, seq_len):
    if self.network_type == 'discrete':
      return RSSMDiscreteState(
        seq_to_batch(rssm_state.logit[:seq_len]),
        seq_to_batch(rssm_state.stoch[:seq_len]),
        seq_to_batch(rssm_state.deter[:seq_len])
      )
    elif self.network_type == 'continuous':
      return RSSMContinuousState(
        seq_to_batch(rssm_state.mean[:seq_len]),
        seq_to_batch(rssm_state.std[:seq_len]),
        seq_to_batch(rssm_state.stoch[:seq_len]),
        seq_to_batch(rssm_state.deter[:seq_len])
      )
      
  def state_batch_to_seq(self, rssm_state, batch_size, seq_len):
    if self.network_type == 'discrete':
      return RSSMDiscreteState(
        batch_to_seq(rssm_state.logit, batch_size, seq_len),
        batch_to_seq(rssm_state.stoch, batch_size, seq_len),
        batch_to_seq(rssm_state.deter, batch_size, seq_len)
      )
    elif self.network_type == 'continuous':
      return RSSMContinuousState(
        batch_to_seq(rssm_state.mean, batch_size, seq_len),
        batch_to_seq(rssm_state.std, batch_size, seq_len),
        batch_to_seq(rssm_state.stoch, batch_size, seq_len),
        batch_to_seq(rssm_state.deter, batch_size, seq_len)
      )
      
def seq_to_batch(sequence_data):
  """
  converts a sequence of length L and batch_size B to a single batch of size L*B
  """
  shape = sequence_data.shape
  batch_data = torch.reshape(sequence_data, [shape[0]*shape[1], *shape[2:]])
  return batch_data

def batch_to_seq(batch_data, batch_size, seq_len):
  """
  converts a single batch of size L*B to a sequence of length L and batch_size B
  """
  shape = batch_data.shape
  sequence_data = torch.reshape(batch_data, [seq_len, batch_size, *shape[1:]])
  return sequence_data