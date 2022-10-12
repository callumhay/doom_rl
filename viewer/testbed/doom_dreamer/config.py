
import os
from typing import Tuple, Dict
from dataclasses import dataclass, field

import torch.nn as nn

CHECKPOINT_DIR = "./checkpoints"

DISCRETE_NET_TYPE   = 'discrete'
CONTINUOUS_NET_TYPE = 'continuous'

NORMAL_DIST_TYPE = 'normal'
BINARY_DIST_TYPE = 'binary'

@dataclass
class Config():
  observation_shape: Tuple
  action_size: int
  
  csv_filepath: str = os.path.join(CHECKPOINT_DIR, "default_log.csv")
  def model_savepath_str(self, train_steps:int) -> str: 
    return os.path.join(CHECKPOINT_DIR, f"doom_model_{train_steps}.chkpt")
  
  capacity: int   = int(8e5)
  seq_len: int    = 50
  batch_size: int = 32
  
  explore_steps: int     = 5000  # Number of steps before training starts, used to seed the replay memory
  train_episodes: int    = 10000 # Number of episodes to train for
  train_every_steps: int = 50    # Number of steps to take between training the network
  save_every_steps: int  = 5e4   # Number of steps to take between saves
  collect_intervals: int = 5     # Number of training runs to take at every train_step for collecting metrics data
  
  use_slow_target: float = True
  slow_target_update_steps: int = 100 # Number of steps to take between updates to the target network
  slow_target_fraction: float = 1.00
  
  horizon: int = 10
  discount: float = 0.999
  td_lambda: float = 0.95
  
  kl_alpha: float = 0.8
  kl_loss_multiplier: float = 0.1
  #obs_kl_loss_multiplier: float = 0.00001
  discount_loss_multiplier: float = 5.0
  actor_entropy_scale: float = 1e-3
  grad_clip: float = 100.0
  
  epsilon_info: Dict = field(default_factory=lambda:{
    'start_epsilon': 1.0, 'epsilon_decay_multiplier': 0.9999998, 'min_epsilon': 0.05
  })
  lr_info: Dict = field(default_factory=lambda:{
    'model':2e-4, 'actor':4e-5, 'critic':1e-4
  })
  
  rssm_info: Dict = field(default_factory=lambda:{
    'type': DISCRETE_NET_TYPE, 'hidden_size': 1280, 
    'deter_size':1280, 'stoch_size':768, 'class_size':32, 
    'category_size':32, 'min_std':0.1, 'activation_fn': nn.ELU
  })
  actor_info: Dict = field(default_factory=lambda:{
    'num_hidden_layers': 3, 'hidden_size': 896, 'activation_fn': nn.ELU
  })
  pos_ori_info: Dict = field(default_factory=lambda:{
    'num_hidden_layers': 2, 'hidden_size': 512, 'activation_fn': nn.ELU,
    'distribution_type': NORMAL_DIST_TYPE
  })
  reward_info: Dict = field(default_factory=lambda:{
    'num_hidden_layers': 3, 'hidden_size': 896, 'activation_fn': nn.ELU, 
    'distribution_type': NORMAL_DIST_TYPE
  })
  critic_info: Dict = field(default_factory=lambda:{
    'num_hidden_layers': 3, 'hidden_size': 896, 'activation_fn': nn.ELU, 
    'distribution_type': NORMAL_DIST_TYPE
  })
  discount_info: Dict = field(default_factory=lambda:{
    'num_hidden_layers': 3, 'hidden_size': 512, 'activation_fn': nn.ELU, 
    'distribution_type': BINARY_DIST_TYPE
  })
  
  encoder_decoder_config: Dict = field(default_factory=lambda:{
    'z_channels': 3, 'ch': 32, 'ch_mult': [ 1,2,2,2 ],
    'num_res_blocks': 1, 'dropout': 0.0,
  })

  @property
  def stoch_size(self):
    if self.rssm_info['type'] == CONTINUOUS_NET_TYPE:
      stoch_size = self.rssm_info['stoch_size']
    elif self.rssm_info['type'] == DISCRETE_NET_TYPE:
      stoch_size = self.rssm_info['category_size'] * self.rssm_info['class_size']
    return stoch_size
  
  @property
  def deter_size(self):
    return self.rssm_info['deter_size']
  
  @property
  def modelstate_size(self):
    return self.stoch_size + self.deter_size