from typing import Tuple, Dict
import torch 
import torch.nn as nn
import torch.distributions as td
import numpy as np


class DoomDiscreteActionModel(nn.Module):
  def __init__(
    self, action_size: int, deter_size: int, stoch_size: int, actor_info: Dict, epsilon_info: Dict
  ) -> None:
    super().__init__()
    
    self.action_size = action_size
    
    self.epsilon = epsilon_info['start_epsilon']
    self.epsilon_decay_multiplier = epsilon_info['epsilon_decay_multiplier']
    self.min_epsilon = epsilon_info['min_epsilon']
    
    assert self.epsilon >= 0 and self.epsilon <= 1
    assert self.min_epsilon >= 0 and self.min_epsilon <= 1
    assert self.epsilon_decay_multiplier >= 0 and self.epsilon_decay_multiplier <= 1
    
    hidden_size = actor_info['hidden_size']
    activation_fn = actor_info['activation_fn']
    num_hidden_layers = actor_info['num_hidden_layers']
    
    model = [nn.Linear(deter_size + stoch_size, hidden_size)]
    model += [activation_fn()]
    for _ in range(num_hidden_layers):
      model += [nn.Linear(hidden_size, hidden_size)]
      model += [activation_fn()]
    model += [nn.Linear(hidden_size, action_size)]
    self.model = nn.Sequential(*model)
    
  def forward(self, modelstate:torch.Tensor) -> Tuple[torch.Tensor, td.OneHotCategorical]:
    action_distribution = self.get_action_distribution(modelstate)
    action = action_distribution.sample()
    action = action + action_distribution.probs - action_distribution.probs.detach()
    return action, action_distribution
  
  def get_action_distribution(self, modelstate:torch.Tensor) -> td.OneHotCategorical:
    logits = self.model(modelstate)
    return td.OneHotCategorical(logits=logits)
    
  def add_exploration(self, action:torch.Tensor):
    if self.training:
      self.epsilon = max(self.min_epsilon, self.epsilon*self.epsilon_decay_multiplier)
    else:
      self.epsilon = self.min_epsilon # evaluation epsilon is just the min
    
    # Epsilon-greedy policy
    if np.random.uniform(0,1) < self.epsilon:
      idx = torch.randint(0, self.action_size, action.shape[:-1], device=action.device)
      action = torch.zeros_like(action, device=action.device)
      action[:,idx] = 1
      
    return action

class DoomDenseModel(nn.Module):
  def __init__(self, output_shape: int, input_size: int, info: Dict, end_activation=None) -> None:
    super().__init__()
    self.output_shape = output_shape
    self.distribution_type = info['distribution_type']
    
    num_hidden_layers = info['num_hidden_layers']
    hidden_size       = info['hidden_size']
    activation_fn     = info['activation_fn']
    
    model = [nn.Linear(input_size, hidden_size)]
    model += [activation_fn()]
    for _ in range(num_hidden_layers):
        model += [nn.Linear(hidden_size, hidden_size)]
        model += [activation_fn()]
    model += [nn.Linear(hidden_size, int(np.prod(self.output_shape)))]
    if end_activation != None:
      model += [end_activation()]
    self.model = nn.Sequential(*model)
    
  def forward(self, x:torch.Tensor) -> torch.Tensor:
    return self.model(x)
    
  def get_distribution(self, input:torch.Tensor) -> td.Independent:
    distributed_inputs = self(input)
    if self.distribution_type == 'normal':
      return td.Independent(td.Normal(distributed_inputs, 1), len(self.output_shape))
    elif self.distribution_type == 'binary':
      return td.Independent(td.Bernoulli(logits=distributed_inputs), len(self.output_shape))
    elif self.distribution_type == None:
      return distributed_inputs
    
    raise NotImplementedError(f"Distribution not implemented for type '{self.distribution_type}'")