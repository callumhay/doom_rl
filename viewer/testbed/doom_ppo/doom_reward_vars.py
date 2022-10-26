
from typing import List, Callable
import vizdoom as vzd
import numpy as np


class DoomReward(object):
  
  def __init__(self, reward_func) -> None:
    self.reward_func = reward_func
  
  def reinit(self, game: vzd.DoomGame) -> None:
    pass
  
  def update_and_calc_reward(self, game: vzd.DoomGame) -> float:
    reward = self.reward_func(game)
    return reward

class DoomRewardVar(object):
  
  def __init__(self, var_types: vzd.GameVariable|List[vzd.GameVariable], reward_func: Callable[[float,float],float]) -> None:
    self.var_types = var_types if isinstance(var_types, list) else [var_types]
    self.reward_func = reward_func
    self.curr_values = [0.0] * len(self.var_types)
    
  def reinit(self, game: vzd.DoomGame) -> None:
    for i, var in enumerate(self.var_types):
      self.curr_values[i] = game.get_game_variable(var)
    
  def update_and_calc_reward(self, game: vzd.DoomGame) -> float:
    reward = 0.0
    for i, var in enumerate(self.var_types):
      new_value = game.get_game_variable(var)
      if new_value != self.curr_values[i]:
        reward += self.reward_func(self.curr_values[i], new_value)
        self.curr_values[i] = new_value
    return reward

#from gym.wrappers.normalize import RunningMeanStd
class DoomPosRewardVar(object):
  
  def __init__(self) -> None:
    self.curr_max_radius = 0.0
    self.init_xyz = np.array([0.,0.,0.])
    #self.dist_rms = RunningMeanStd(shape=())
    #self.dist_np  = np.zeros((1,))
    
  def _curr_game_xyz(self, game: vzd.DoomGame) -> np.ndarray:
    return np.array([
      game.get_game_variable(vzd.GameVariable.POSITION_X),
      game.get_game_variable(vzd.GameVariable.POSITION_Y),
      game.get_game_variable(vzd.GameVariable.POSITION_Z),
    ])
  
  def reinit(self, game: vzd.DoomGame) -> None:
    self.curr_max_radius = 0.0
    self.init_xyz = self._curr_game_xyz(game)
    
  def update_and_calc_reward(self, game: vzd.DoomGame) -> float:
    reward = 0.0
    curr_xyz = self._curr_game_xyz(game)
    dist = np.sqrt(np.sum((curr_xyz-self.init_xyz)**2))
    
    #self.dist_np[0] = dist
    #self.dist_rms.update(self.dist_np)
    
    radius_diff = dist - self.curr_max_radius
    if radius_diff > 0:
      reward += 0.1 * radius_diff
      self.curr_max_radius = dist

    # TODO: Punish for oscillating in the same area for a long time?

    return reward

