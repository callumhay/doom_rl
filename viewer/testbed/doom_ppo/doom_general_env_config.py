
import gym
import vizdoom as vzd
import numpy as np
from doom_reward_vars import DoomReward, DoomRewardVar, DoomAdvancedPosRewardVar

_living_reward = -0.01
_kill_reward   = 5.0
_death_penalty  = 5.0
_map_completed_reward = 100.0
_timeout_reward = 0 # NOTE: Having a timeout reward is detrimental to convergence!
_episode_timeout = 9999999

# Custom config class for the vizdoomenv
class DoomGeneralEnvConfig(object):
  
  def __init__(self, map:str) -> None:
    self.map = map
    self.reward_vars = []
  
  def load(self, env) -> None:
    game = env.game
    
    game.set_doom_map(self.map)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_episode_timeout(_episode_timeout)
    game.set_render_hud(False)
    #game.set_render_minimal_hud(True)
    game.set_render_decals(True)
    game.set_render_particles(True)
    game.set_render_effects_sprites(True)
    game.set_render_corpses(True)
    game.set_render_messages(False)
    
    game.clear_available_buttons()
    game.set_available_buttons([
      vzd.Button.MOVE_LEFT,
      vzd.Button.MOVE_RIGHT,
      vzd.Button.TURN_LEFT,
      vzd.Button.TURN_RIGHT,
      vzd.Button.ATTACK,
      vzd.Button.MOVE_FORWARD,
      vzd.Button.MOVE_BACKWARD,
      vzd.Button.USE,
      vzd.Button.SPEED
    ])
    
    ammo_vars = [
      vzd.GameVariable.AMMO2, # Amount of ammo for the pistol
      vzd.GameVariable.AMMO3, # ...shotgun
      vzd.GameVariable.AMMO4, # etc.
      vzd.GameVariable.AMMO5,
      vzd.GameVariable.AMMO6,
      vzd.GameVariable.AMMO7,
      vzd.GameVariable.AMMO8,
      vzd.GameVariable.AMMO9,
    ]
    
    # Adds game variables that will be included in state (for calculating rewards)
    game.clear_available_game_variables()
    game.add_available_game_variable(vzd.GameVariable.ANGLE)       # Player orientation angle
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Z)
    game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)   # Counts the number of monsters killed during the current episode. 
    game.add_available_game_variable(vzd.GameVariable.DAMAGECOUNT) # Counts the damage dealt to monsters/players/bots during the current episode.
    game.add_available_game_variable(vzd.GameVariable.SECRETCOUNT) # Counts the number of secret location/objects discovered during the current episode.
    game.add_available_game_variable(vzd.GameVariable.HEALTH)      # Current player health
    game.add_available_game_variable(vzd.GameVariable.ARMOR)       # Current player armor
    game.add_available_game_variable(vzd.GameVariable.ITEMCOUNT)   # Item count (pick-ups)
    game.add_available_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
    for ammo_var in ammo_vars:
      game.add_available_game_variable(ammo_var)
    #TODO: Other variables... ?
    
    # Reward functions (how we calculate the reward when specific game variables change)
    health_reward_func     = lambda oldHealth, newHealth: (0.2 if newHealth > oldHealth else 0.1) * (newHealth-oldHealth)
    armor_reward_func      = lambda oldArmor, newArmor: (0.1 if newArmor > oldArmor else 0.0) * (newArmor-oldArmor)
    item_reward_func       = lambda oldItemCount, newItemCount: max(0.0, 0.1*(newItemCount-oldItemCount))
    secrets_reward_func    = lambda oldNumSecrets, newNumSecrets: max(0.0, newNumSecrets-oldNumSecrets)
    dmg_reward_func        = lambda oldDmg, newDmg: max(0.0, 0.01*(newDmg-oldDmg))
    kill_count_reward_func = lambda oldKillCount, newKillCount: max(0, _kill_reward*(newKillCount-oldKillCount))
    ammo_reward_func       = lambda oldAmmo, newAmmo: (0.1 if newAmmo > oldAmmo else 0.05) * (newAmmo-oldAmmo)
    
    map_complete_reward_func = lambda _: _map_completed_reward if env.is_map_ended() else 0.0
    timeout_reward_func      = lambda _: _timeout_reward if not env.is_map_ended() and env.is_episode_finished() else 0.0
    game.set_death_penalty(_death_penalty)
    game.set_living_reward(_living_reward)
    
    self.reward_vars = [
      DoomReward(map_complete_reward_func),
      DoomReward(timeout_reward_func),
      DoomRewardVar(vzd.GameVariable.KILLCOUNT, kill_count_reward_func),
      DoomRewardVar(vzd.GameVariable.DAMAGECOUNT, dmg_reward_func),
      DoomRewardVar(vzd.GameVariable.SECRETCOUNT, secrets_reward_func),
      DoomRewardVar(vzd.GameVariable.HEALTH, health_reward_func),
      DoomRewardVar(vzd.GameVariable.ARMOR, armor_reward_func),
      DoomRewardVar(vzd.GameVariable.ITEMCOUNT, item_reward_func),
      DoomRewardVar(ammo_vars, ammo_reward_func),
      DoomAdvancedPosRewardVar(),
    ]
    
  def game_variable_spaces(self):
    spaces = []
    spaces.append(gym.spaces.Box(0.0, 1.0, (3,)))
    return spaces
  
  def reset(self, game):
    for reward_var in self.reward_vars:
      reward_var.reinit(game)
  
  def step_reward(self, game):
    reward = 0.0
    for reward_var in self.reward_vars: 
      reward += reward_var.update_and_calc_reward(game)
    return reward
  
  def game_variable_observations(self, game):
    obs = []

    #max_excl_pos   = 101 
    obs.append(np.array([
      #game.get_game_variable(vzd.GameVariable.POSITION_X) % max_excl_pos,
      #game.get_game_variable(vzd.GameVariable.POSITION_Y) % max_excl_pos,
      #game.get_game_variable(vzd.GameVariable.POSITION_Z) % max_excl_pos,
      (game.get_game_variable(vzd.GameVariable.ANGLE) / 360.0) % 1.0,
      min(100, game.get_game_variable(vzd.GameVariable.HEALTH)) / 100.0,
      min(100, game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)) / 100.0,
    ]))

    return obs
