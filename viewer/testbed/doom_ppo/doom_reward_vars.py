
from typing import Callable
import vizdoom as vzd
import numpy as np

class DoomRewardVar(object):
  
  def __init__(self, var_type: vzd.GameVariable, desc: str, reward_func: Callable[[float,float],float]) -> None:
    self.var_type = var_type
    self.desc = desc
    self.reward_func = reward_func
    self.curr_value = 0.0
    
  def reinit(self, game: vzd.DoomGame) -> None:
    self.curr_value = game.get_game_variable(self.var_type)
    
  def update_and_calc_reward(self, game:vzd.DoomGame) -> float:
    new_value = game.get_game_variable(self.var_type)
    reward = 0.0
    if new_value != self.curr_value:
      reward = self.reward_func(self.curr_value, new_value)
      self.curr_value = new_value
    return reward
  
  
class DoomPosRewardVar(object):
  
  def __init__(self) -> None:
    self.desc = "Player position"
    self.curr_max_radius = 0.0
    self.init_xyz = np.array([0.,0.,0.])
  
  def _curr_game_xyz(self, game: vzd.DoomGame) -> np.ndarray:
    return np.array([
      game.get_game_variable(vzd.GameVariable.POSITION_X),
      game.get_game_variable(vzd.GameVariable.POSITION_Y),
      game.get_game_variable(vzd.GameVariable.POSITION_Z),
    ])
  
  def reinit(self, game: vzd.DoomGame) -> None:
    self.curr_max_radius = 0.0
    self.init_xyz = self._curr_game_xyz(game)
    
  def update_and_calc_reward(self, game:vzd.DoomGame) -> float:
    reward = 0.0
    curr_xyz = self._curr_game_xyz(game)
    dist = np.sqrt(np.sum((curr_xyz-self.init_xyz)**2))
    
    radius_diff = dist - self.curr_max_radius
    if radius_diff > 0:
      reward += 0.02 * radius_diff
      self.curr_max_radius = dist
    else:
      reward -= 0.01 # Smaller punishment for not exploring
      
    return reward
    