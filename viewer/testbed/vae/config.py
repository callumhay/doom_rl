
from typing import Tuple, Dict
from dataclasses import dataclass, field

import torch.nn as nn

DISCRETE_NET_TYPE   = 'discrete'
CONTINUOUS_NET_TYPE = 'continuous'

NORMAL_DIST_TYPE = 'normal'
BINARY_DIST_TYPE = 'binary'

class Config():
  capacity: int   = int(1e6)
  seq_len: int    = 50
  batch_size: int = 50
  explore_steps: int = 5000 # Number of steps before training starts, used to seed the replay memory
  
  observation_shape: Tuple
  action_size: int
  embedding_size: int
  
  horizon: int = 10
  discount: float = 0.99
  td_lambda: float = 0.99
  
  kl_alpha: float = 0.8
  kl_loss_multiplier: float = 0.1
  discount_loss_multiplier: float = 5.0
  actor_entropy_scale: float = 1e-3
  
  
  epsilon_info: Dict = field(default_factory=lambda:{
    'start_epsilon': 1.0, 'epsilon_decay_multiplier': 0.9999998, 'min_epsilon': 0.05
  })
  lr_info: Dict = field(default_factory=lambda:{
    'model':2e-4, 'actor':4e-5, 'critic':1e-4
  })
  
  
  rssm_info: Dict = field(default_factory=lambda:{
    'type': DISCRETE_NET_TYPE, 'hidden_size': 1024, 
    'deter_size':2048, 'stoch_size':1024, 'class_size':128, 
    'category_size':128, 'min_std':0.1
  })
  actor_info: Dict = field(default_factory=lambda:{
    'num_hidden_layers': 3, 'hidden_size': 1024, 'activation_fn': nn.ELU
  })
  reward_info: Dict = field(default_factory=lambda:{
    'num_hidden_layers': 3, 'hidden_size': 1024, 'activation_fn': nn.ELU, 
    'distribution_type': NORMAL_DIST_TYPE
  })
  critic_info: Dict = field(default_factory=lambda:{
    'num_hidden_layers': 3, 'hidden_size': 1024, 'activation_fn': nn.ELU, 
    'distribution_type': NORMAL_DIST_TYPE
  })
  discount_info: Dict = field(default_factory=lambda:{
    'num_hidden_layers': 3, 'hidden_size': 1024, 'activation_fn': nn.ELU, 
    'distribution_type': BINARY_DIST_TYPE
  })