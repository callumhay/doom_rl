import vizdoom as vzd
from random import choice

class DoomEnv:
  def __init__(self, map:str) -> None:
    self.map = map
    self.episode_timeout = 999999999
    self.frameskip = 4
    
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
    #game.set_render_messages(False)
    
    self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X256)

    # Set cv2 friendly format.
    self.game.set_screen_format(vzd.ScreenFormat.RGB24)

    # Enables labeling of the in game objects.
    self.game.set_labels_buffer_enabled(True)

    self.game.clear_available_game_variables()
    self.game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    self.game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    self.game.add_available_game_variable(vzd.GameVariable.POSITION_Z)
    
    self.actions = {
      "MoveLeft"    : [True,False,False,False,False,False,False,False],
      "MoveRight"   : [False,True,False,False,False,False,False,False],
      "TurnLeft"    : [False,False,True,False,False,False,False,False],
      "TurnRight"   : [False,False,False,True,False,False,False,False],
      "Attack"      : [False,False,False,False,True,False,False,False],
      "MoveForward" : [False,False,False,False,False,True,False,False],
      "MoveBackward": [False,False,False,False,False,False,True,False],
      "Use"         : [False,False,False,False,False,False,False,True],
      "NoAction"    : [False,False,False,False,False,False,False,False],
    }
    self.game.set_available_buttons([
      vzd.Button.MOVE_LEFT,
      vzd.Button.MOVE_RIGHT,
      vzd.Button.TURN_LEFT,
      vzd.Button.TURN_RIGHT,
      vzd.Button.ATTACK,
      vzd.Button.MOVE_FORWARD,
      vzd.Button.MOVE_BACKWARD,
      vzd.Button.USE
    ])
    self.game.set_window_visible(True)
    self.game.init()

  def is_episode_finished(self):
    return self.game.get_state() == None or self.game.is_episode_finished()
  
  def is_map_ended(self):
    return self.game.is_episode_finished() and not self.game.is_player_dead() and self.game.get_episode_time() < self.episode_timeout

  def random_action(self):
    return choice(list(self.actions.values()))
  
  def reset(self) -> vzd.vizdoom.GameState:
    if not self.game.is_running():
      return None

    #self.game.set_doom_map(self.map)
    self.game.new_episode()
    return self.game.get_state()

  def step(self, action:str|list[bool]) -> list[vzd.vizdoom.GameState,float,bool]:
    state = self.game.get_state()
    reward = 0.0
    actionVec = self.actions[action] if isinstance(action, str) else action
    assert isinstance(actionVec, list)
    
    if any(actionVec):
      try:
        self.game.make_action(actionVec, self.frameskip)
      except:
        exit(0)
    
    # TODO: Implement game variables for rewards
    #if state != None: for game_var in self.game_reward_vars : reward += game_var.update_and_calc_reward(self.game)
    
    if self.is_map_ended():
      print("Map was completed, nice!")
      reward += 1000
    if self.game.is_player_dead():
      print("Agent died!")
      reward -= 20
    
    return [state, reward, self.is_episode_finished()] 
