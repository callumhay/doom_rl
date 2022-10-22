import os
from typing import Tuple

import torchvision.transforms.functional as torchvisfunc
import numpy as np
import scipy.ndimage
import vizdoom as vzd
import random

from doom_reward_vars import DoomReward, DoomRewardVar, DoomPosRewardVar

DEFAULT_PREPROCESS_FINAL_SHAPE = (6, 75, 100)
NUM_LABEL_CLASSES = 256

_VIZDOOM_SCENARIO_PATH = "_vizdoom"
_frameskip = 4
_living_reward = 0.0 #-0.01 / _frameskip
_kill_reward   = 5.0
_death_reward  = -5.0
_map_completed_reward = 10.0


_DOOM_BAD_STUFF_SET = set([
  'ShotgunGuy','ChaingunGuy','BaronOfHell','Zombieman','DoomImp','Arachnotron','SpiderMastermind',
  'Demon','Spectre','DoomImpBall','Cacodemon','Revenant','RevenantTracer',
  'StealthArachnotron','StealthArchvile','StealthCacodemon',
  'StealthChaingunGuy','StealthDemon','StealthDoomImp','StealthFatso','StealthRevenant',
  'CacodemonBall','PainElemental','ArchvileFire',
  'StealthBaron','StealthHellKnight','StealthZombieMan','StealthShotgunGuy',
  'LostSoul','Archvile','Fatso','HellKnight','Cyberdemon','ArachnotronPlasma','BaronBall','FatShot'
])
_DOOM_GOOD_STUFF_SET = set([
  'Stimpack', 'Medikit', 'Soulsphere', 'GreenArmor', 'BlueArmor', 'ArmorBonus',
  'Megasphere', 'InvulnerabilitySphere', 'BlurSphere', 'Backpack', 'HealthBonus',
  'RadSuit', 'BlueCard', 'RedCard', 'YellowCard', 'YellowSkull', 'RedSkull', 'BlueSkull',
])
_DOOM_WEAPON_STUFF_SET = set([
  'Clip', 'Shell', 'Cell', 'ClipBox', 'RocketAmmo', 'RocketBox', 'CellPack', 'ShellBox',
  'Shotgun','Chaingun','RocketLauncher','PlasmaRifle','BFG9000','Chainsaw','SuperShotgun',
])

def _get_label_type_id(label):
  name = label.object_name
  if name in _DOOM_BAD_STUFF_SET: return 1
  elif name in _DOOM_GOOD_STUFF_SET: return 2
  elif name in _DOOM_WEAPON_STUFF_SET: return 3
  return None

class DoomEnv(object):
  def __init__(self, args, episode_max_steps:int, window_visible=True, 
               screen_res=vzd.ScreenResolution.RES_200X150, 
               downsample_res_h=DEFAULT_PREPROCESS_FINAL_SHAPE[1], 
               downsample_res_w=DEFAULT_PREPROCESS_FINAL_SHAPE[2]) -> None:
    
    self.use_labels_buffer = args.use_labels_buffer
    self.episode_timeout = episode_max_steps
    self.clip_reward = args.clip_reward
    self.preprocess_res = (downsample_res_h, downsample_res_w)
    self.preprocess_final_shape = (6 if self.use_labels_buffer else 3, downsample_res_h, downsample_res_w)

    self.game = vzd.DoomGame()
    
    # Be sure to set these things before setting the scenario
    # (so they can be overwritten by the scenario when necessary)
    self.game.set_living_reward(_living_reward)
    self.game.set_episode_timeout(self.episode_timeout)
    
    self.game.set_doom_game_path(os.path.join(_VIZDOOM_SCENARIO_PATH, "doom.wad"))#"../../../build/bin/doom.wad")
    self.game.set_doom_scenario_path(os.path.join(_VIZDOOM_SCENARIO_PATH, args.scenario_name) + ".wad")#"../../../build/bin/doom.wad")
    config_path = os.path.join(_VIZDOOM_SCENARIO_PATH, args.scenario_name) + ".cfg"
    if os.path.exists(config_path): 
      self.using_config = True
      self.game.load_config(config_path)
      
    self.game.set_doom_map(args.map)
    self.game.set_mode(vzd.Mode.PLAYER)
    self.game.set_episode_start_time(10)
    
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

    self.game.set_screen_resolution(screen_res) #200x150 -> 100x75
    self.game.set_screen_format(vzd.ScreenFormat.RGB24)
    self.game.set_depth_buffer_enabled(False) # Only using the colour buffer
    if self.use_labels_buffer:
      self.game.set_labels_buffer_enabled(True) # Enables labeling of the in game objects.

    self.actions = {
      #"MoveLeft"    : np.array([True,False,False,False,False,False,False,False]),
      #"MoveRight"   : np.array([False,True,False,False,False,False,False,False]),
      "TurnLeft"    : np.array([True,False,False,False,False,False,True]),
      "TurnRight"   : np.array([False,True,False,False,False,False,True]),
      "Attack"      : np.array([False,False,True,False,False,False,True]),
      "MoveForward" : np.array([False,False,False,True,False,False,True]),
      "MoveBackward": np.array([False,False,False,False,True,False,True]),
      "Use"         : np.array([False,False,False,False,False,True,True]),
      
      "NoAction"    : np.array([False,False,False,False,False,False,False]),
    }
    self.avail_action_keys = [k for k in self.actions.keys() if k != "NoAction"]
    self.game.clear_available_buttons()
    self.game.set_available_buttons([
      #vzd.Button.MOVE_LEFT,
      #vzd.Button.MOVE_RIGHT,
      vzd.Button.TURN_LEFT,
      vzd.Button.TURN_RIGHT,
      vzd.Button.ATTACK,
      vzd.Button.MOVE_FORWARD,
      vzd.Button.MOVE_BACKWARD,
      vzd.Button.USE,
      vzd.Button.SPEED
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
    health_reward_func     = lambda oldHealth, newHealth: (0.02 if newHealth > oldHealth else 0.01) * (newHealth-oldHealth)
    armor_reward_func      = lambda oldArmor, newArmor: (0.01 if newArmor > oldArmor else 0.0) * (newArmor-oldArmor)
    item_reward_func       = lambda oldItemCount, newItemCount: 0.01*(newItemCount-oldItemCount) if newItemCount > oldItemCount else 0
    secrets_reward_func    = lambda oldNumSecrets, newNumSecrets: 1.0 if newNumSecrets > oldNumSecrets else 0.0
    dmg_reward_func        = lambda oldDmg, newDmg: 0.01*(newDmg-oldDmg) if newDmg > oldDmg else 0.0
    kill_count_reward_func = lambda oldKillCount, newKillCount: _kill_reward*(newKillCount-oldKillCount) if newKillCount > oldKillCount else 0.0
    ammo_reward_func       = lambda oldAmmo, newAmmo: (0.02 if newAmmo > oldAmmo else 0.1) * (newAmmo-oldAmmo)
    
    map_complete_reward_func = lambda env: _map_completed_reward if env.is_map_ended() else 0.0
    death_reward_func        = lambda env: _death_reward if env.game.is_player_dead() else 0.0
    
    if self.using_config:
      self.reward_vars = []
    else:
      self.reward_vars = [
        DoomReward(map_complete_reward_func),
        DoomReward(death_reward_func),
        DoomRewardVar(vzd.GameVariable.KILLCOUNT, "Kill count", kill_count_reward_func),
        DoomRewardVar(vzd.GameVariable.DAMAGECOUNT, "Monster/Environment damage", dmg_reward_func),
        DoomRewardVar(vzd.GameVariable.SECRETCOUNT, "Secrets found", secrets_reward_func),
        DoomRewardVar(vzd.GameVariable.HEALTH, "Player health", health_reward_func),
        DoomRewardVar(vzd.GameVariable.ARMOR, "Player armor", armor_reward_func),
        DoomRewardVar(vzd.GameVariable.ITEMCOUNT, "Item count", item_reward_func),
        DoomRewardVar(vzd.GameVariable.AMMO2, "Pistol ammo count", ammo_reward_func),
        DoomPosRewardVar(),
      ]
    
    self.next_step_reset = True
    self.curr_ep_return = 0.0
    self.curr_ep_len = 0
    
    self.game.set_window_visible(window_visible)
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
    return turns

  def is_episode_finished(self):
    return self.game.get_state() == None or self.game.is_episode_finished()
  
  def is_map_ended(self):
    return self.game.is_episode_finished() and not self.game.is_player_dead() and self.game.get_episode_time() < self.episode_timeout

  def random_action(self) -> np.ndarray:
    return self.actions[random.choice(self.avail_action_keys)]
  
  def close(self):
    self.game.close()
    
  def labels_array(self, game_state):
    labels = np.zeros((NUM_LABEL_CLASSES))
    label_inds = [l.value for l in game_state.labels]
    labels[label_inds] = 1
    return labels
  
  def labels_buffer(self, game_state):
    _mapping = self.labels_array(game_state).astype(np.uint8)
    labels_buf = (_mapping[game_state.labels_buffer] == 1).astype(np.uint8)
    return labels_buf
 
  # Create label buffers for good, bad and utility pixels on screen
  # Returns a buffer of shape (3, unscaled_H, unscaled_W)
  def labels_buffers(self, game_state):
    _mapping = np.zeros((256,), dtype=np.uint8)
    for label in game_state.labels:
      type_id = _get_label_type_id(label)
      if type_id is not None:
        _mapping[label.value] = type_id
    labels_bufs = -(_mapping[game_state.labels_buffer] == np.arange(1,4)[:,None,None]).astype(np.uint8) / 255.0
    return labels_bufs
 
  def get_observation(self, game_state):
    obs_buf = self.preprocess_screenbuffer(game_state.screen_buffer).numpy()
    if self.use_labels_buffer:
      labelsbuf_obs = self.labels_buffers(game_state)
      labelsbuf_obs = scipy.ndimage.zoom(labelsbuf_obs, [1.0, 0.5, 0.5], order=0)
      #labelsbuf_obs = self.labels_buffer(game_state)
      #labelsbuf_obs = np.expand_dims(scipy.ndimage.zoom(labelsbuf_obs, 0.5, order=0),0)
      obs_buf = np.concatenate([obs_buf, labelsbuf_obs], 0)
    return obs_buf
  
  def observation_shape(self):
    return self.preprocess_final_shape
  
  def reset(self) -> np.ndarray:
    if not self.game.is_running():
      return None

    self.game.new_episode()
    state = self.game.get_state()
    
    # Reset all our reward variables
    for rv in self.reward_vars:
      rv.reinit(self.game)
    
    self.next_step_reset = False
    self.curr_ep_return = 0.0
    self.curr_ep_len = 0
    
    observation = self.get_observation(state)
    return observation, self.labels_array(state)

  def step(self, action:np.ndarray) -> Tuple[np.ndarray,float,bool]:
    if self.next_step_reset:
      return *self.reset(), 0.0, False, {}
    
    state = self.game.get_state()
    assert state != None
    reward = 0.0
    
    if not isinstance(action, np.ndarray):
      # Convert the action into the proper array for feeding vizdoom
      action = self.actions[self.avail_action_keys[action]]
    
    if any(action):
      try:
        reward += self.game.make_action(action, _frameskip)
      except:
        exit(0)
    
    # Calculate the sum of all in-game/gameplay rewards
    if state != None: 
      for rv in self.reward_vars: 
        reward += rv.update_and_calc_reward(self)
    
    reward = np.sign(reward) if self.clip_reward else reward
    self.curr_ep_return += reward
    self.curr_ep_len += 1
    
    done = self.is_episode_finished()
    info = {}
    if done:
      self.next_step_reset = True
      info = {"episode": {"r": self.curr_ep_return, "l": self.curr_ep_len}}
    
    observation = self.get_observation(state)
    return observation, self.labels_array(state), reward, done, info


  def preprocess_screenbuffer(self, screenbuf):
    screenbuf = torchvisfunc.to_tensor(screenbuf)
    screenbuf = torchvisfunc.resize(screenbuf, self.preprocess_res)
    return screenbuf
