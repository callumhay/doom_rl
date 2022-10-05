from typing import Tuple

import torch
import torchvision.transforms.functional as torchvisfunc
import numpy as np
import vizdoom as vzd
import random

from doom_reward_vars import DoomRewardVar, DoomPosRewardVar

PREPROCESS_RES_H_W = (64,80) #(128,160) # Must be (H,W)!
PREPROCESS_FINAL_SHAPE_C_H_W = (3, PREPROCESS_RES_H_W[0], PREPROCESS_RES_H_W[1])

_frameskip = 4
_living_reward = -0.01 / _frameskip
_kill_reward  = 10.0
_death_reward = -20.0
_map_completed_reward = 1000.0

class DoomEnv(object):
  def __init__(self, map:str, episode_max_steps:int) -> None:
    self.map = map
    self.episode_timeout = episode_max_steps

    self.game = vzd.DoomGame()
    self.game.set_doom_game_path("../../../build/bin/doom.wad")
    self.game.set_doom_scenario_path("../../../build/bin/doom.wad")
    self.game.set_doom_map(self.map)
    self.game.set_mode(vzd.Mode.PLAYER)
    self.game.set_episode_start_time(10)
    self.game.set_episode_timeout(self.episode_timeout)
    #game.add_game_args("+freelook 1")

    # Use other config file if you wish.
    #game.load_config(args.config)
    self.game.set_render_hud(True)
    self.game.set_render_minimal_hud(False)
    self.game.set_render_decals(True)
    self.game.set_render_particles(True)
    self.game.set_render_effects_sprites(True)
    self.game.set_render_corpses(True)
    self.game.set_render_messages(False)
    self.game.set_living_reward(_living_reward)
    
    self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X256)

    # Set cv2 friendly format.
    self.game.set_screen_format(vzd.ScreenFormat.RGB24)

    # Enables labeling of the in game objects.
    self.game.set_labels_buffer_enabled(True)

    self.actions = {
      #"MoveLeft"    : np.array([True,False,False,False,False,False,False,False]),
      #"MoveRight"   : np.array([False,True,False,False,False,False,False,False]),
      "TurnLeft"    : np.array([True,False,False,False,False,False]),
      "TurnRight"   : np.array([False,True,False,False,False,False]),
      "Attack"      : np.array([False,False,True,False,False,False]),
      "MoveForward" : np.array([False,False,False,True,False,False]),
      "MoveBackward": np.array([False,False,False,False,True,False]),
      "Use"         : np.array([False,False,False,False,False,True]),
      
      "NoAction"    : np.array([False,False,False,False,False,False]),
    }
    self.avail_action_keys = [k for k in self.actions.keys() if k != "NoAction"]
    self.game.set_available_buttons([
      #vzd.Button.MOVE_LEFT,
      #vzd.Button.MOVE_RIGHT,
      vzd.Button.TURN_LEFT,
      vzd.Button.TURN_RIGHT,
      vzd.Button.ATTACK,
      vzd.Button.MOVE_FORWARD,
      vzd.Button.MOVE_BACKWARD,
      vzd.Button.USE
    ])
    
    # Adds game variables that will be included in state (for calculating rewards)
    self.game.clear_available_game_variables()
    self.game.add_available_game_variable(vzd.GameVariable.ANGLE)       # Player orientation angle
    self.game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    self.game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    self.game.add_available_game_variable(vzd.GameVariable.POSITION_Z)
    self.game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)   # Counts the number of monsters killed during the current episode. 
    self.game.add_available_game_variable(vzd.GameVariable.DAMAGECOUNT) # Counts the damage dealt to monsters/players/bots during the current episode.
    self.game.add_available_game_variable(vzd.GameVariable.SECRETCOUNT) # Counts the number of secret location/objects discovered during the current episode.
    self.game.add_available_game_variable(vzd.GameVariable.HEALTH)      # Current player health
    self.game.add_available_game_variable(vzd.GameVariable.ARMOR)       # Current player armor
    self.game.add_available_game_variable(vzd.GameVariable.ITEMCOUNT)   # Item count (pick-ups)
    self.game.add_available_game_variable(vzd.GameVariable.AMMO2)       # Amount of ammo for the pistol
    #TODO: Other ammo variables... AMMO3, AMMO4, ... , AMMO9

    
    # Reward functions (how we calculate the reward when specific game variables change)
    health_reward_func     = lambda oldHealth, newHealth: 0.25 * (newHealth-oldHealth)
    armor_reward_func      = lambda oldArmor, newArmor: (1.0 if newArmor > oldArmor else 0.0) * (newArmor-oldArmor)
    item_reward_func       = lambda oldItemCount, newItemCount:  (newItemCount-oldItemCount) if newItemCount > oldItemCount else 0
    secrets_reward_func    = lambda oldNumSecrets, newNumSecrets: 5 if newNumSecrets > oldNumSecrets else 0
    dmg_reward_func        = lambda oldDmg, newDmg: 0.5*(newDmg-oldDmg) if newDmg > oldDmg else 0
    kill_count_reward_func = lambda oldKillCount, newKillCount: _kill_reward if newKillCount > oldKillCount else 0
    ammo_reward_func       = lambda oldAmmo, newAmmo: (1.0 if newAmmo > oldAmmo else 0.1) * (newAmmo-oldAmmo)
    
    self.reward_vars = [
      DoomRewardVar(vzd.GameVariable.KILLCOUNT, "Kill count", kill_count_reward_func),
      DoomRewardVar(vzd.GameVariable.DAMAGECOUNT, "Monster/Environment damage", dmg_reward_func),
      DoomRewardVar(vzd.GameVariable.SECRETCOUNT, "Secrets found", secrets_reward_func),
      DoomRewardVar(vzd.GameVariable.HEALTH, "Player health", health_reward_func),
      DoomRewardVar(vzd.GameVariable.ARMOR, "Player armor", armor_reward_func),
      DoomRewardVar(vzd.GameVariable.ITEMCOUNT, "Item count", item_reward_func),
      DoomRewardVar(vzd.GameVariable.AMMO2, "Pistol ammo count", ammo_reward_func),
      DoomPosRewardVar(),
    ]
    
    self.game.set_window_visible(True)
    self.game.init()

  @property
  def action_size(self): return len(self.avail_action_keys)

  def player_pos(self, normalize=True):
    pos = np.array([
      self.game.get_game_variable(vzd.GameVariable.POSITION_X),
      self.game.get_game_variable(vzd.GameVariable.POSITION_Y),
      self.game.get_game_variable(vzd.GameVariable.POSITION_Z),
    ])
    if normalize: pos /= 32000.0 # This is somewhat arbitrary, just want to keep the numbers in [0,1], preferably
    return pos
  
  def player_heading(self):
    # Return as a number in [0,1) where 0 is 0 degrees and 1 is 360 degrees
    turns = self.game.get_game_variable(vzd.GameVariable.ANGLE) / 360.0 % 1.0
    # TODO: What is this value?
    return turns

  def is_episode_finished(self):
    return self.game.get_state() == None or self.game.is_episode_finished()
  
  def is_map_ended(self):
    return self.game.is_episode_finished() and not self.game.is_player_dead() and self.game.get_episode_time() < self.episode_timeout

  def random_action(self) -> np.ndarray:
    return self.actions[random.choice(self.avail_action_keys)]
  
  def reset(self) -> np.ndarray:
    if not self.game.is_running():
      return None

    #self.game.set_doom_map(self.map)
    self.game.new_episode()
    state = self.game.get_state()
    
    # Reset all our reward variables
    for rv in self.reward_vars:
      rv.reinit(self.game)
    
    observation = DoomEnv.preprocess_screenbuffer(state.screen_buffer).numpy()
    return observation

  def step(self, action:np.ndarray) -> Tuple[np.ndarray,float,bool]:
    state = self.game.get_state()
    reward = 0.0
    assert isinstance(action, np.ndarray)
    
    if any(action):
      try:
        reward += self.game.make_action(action, _frameskip)
      except:
        exit(0)
    
    # Calculate the sum of all in-game/gameplay rewards
    if state != None: 
      for rv in self.reward_vars: 
        reward += rv.update_and_calc_reward(self.game)
    
    if self.is_map_ended():
      print("Map was completed, nice!")
      reward += _map_completed_reward
    if self.game.is_player_dead():
      #print("Agent died!")
      reward += _death_reward
      
    if state == None:
      print("Game state was 'None', returning empty, terminal state.")
      return torch.zeros(PREPROCESS_FINAL_SHAPE_C_H_W), reward, True
    
    observation = DoomEnv.preprocess_screenbuffer(state.screen_buffer).numpy()
    return observation, reward, self.is_episode_finished()


  def preprocess_screenbuffer(screenbuf):
    screenbuf = torchvisfunc.to_tensor(screenbuf)
    screenbuf = torchvisfunc.resize(screenbuf, PREPROCESS_RES_H_W)
    #screenbuf = torchvisfunc.normalize(screenbuf, (0.485,0.456,0.406), (0.229,0.224,0.225))

    return screenbuf
  
  def deprocess_screenbuffer(screenbuf_tensor):
    #screenbuf_tensor = torchvisfunc.normalize(screenbuf_tensor, (0.,0.,0.), (1.0/0.229,1.0/0.224,1.0/0.225))
    #screenbuf_tensor = torchvisfunc.normalize(screenbuf_tensor, (-0.485,-0.456,-0.406), (1.,1.,1.))
    return screenbuf_tensor