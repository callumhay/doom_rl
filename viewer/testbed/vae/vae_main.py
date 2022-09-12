#!/usr/bin/env python3

from cmath import inf
import os
import vizdoom as vzd
import torch
import torchvision.transforms.functional as torchvisfunc
import torchshow as ts
import argparse
from random import choice
import numpy as np
from torchsummary import summary

from doom_vae import DoomVAE

modelFilepath = "./checkpoints/doom_vae.model"

DEVICE_STR = 'cpu'
DEVICE = torch.device(DEVICE_STR)

latent_space_size = 256
preprocess_res = (128,160) # Must be (H,W)!
#non_hud_height = 83
preprocessed_final_shape = (3, preprocess_res[0], preprocess_res[1])#(3, non_hud_height, preprocess_res[1])

def preprocess_screenbuffer(screenbuf: np.ndarray) -> torch.Tensor:
  screenbuf = torchvisfunc.to_tensor(screenbuf)
  screenbuf = torchvisfunc.resize(screenbuf, preprocess_res)
  screenbuf = torchvisfunc.normalize(screenbuf, (0.485,0.456,0.406), (0.229,0.224,0.225))
  #screenbuf = screenbuf[:,0:non_hud_height,:]
  #ts.show(screenbuf)
  assert screenbuf.shape == (preprocessed_final_shape[0],preprocessed_final_shape[1],preprocessed_final_shape[2])
  return screenbuf


if __name__ =="__main__":
  parser = argparse.ArgumentParser(description="Train a Doom VAE network.")
  parser.add_argument('-net', '--net', '--n', '-n', help="An existing VAE model network to load.", default=modelFilepath)
  parser.add_argument('-map', '--map', '--m', '-m', help="The Doom map to load.", default="E1M1")
  args = parser.parse_args()
  
  if torch.cuda.is_available():
    DEVICE_STR = 'cuda'
    DEVICE = torch.device(DEVICE_STR)
    #torch.backends.cudnn.benchmark = True   

  vaeNet = DoomVAE(preprocessed_final_shape, latent_space_size)
  if len(args.net) > 0:
    if os.path.exists(args.net):
      vaeNet.load_state_dict(torch.load(args.net))
      print("Model loaded from " + args.net)
    else:
      print("Could not find file " + args.net)

  vaeNet = vaeNet.to(DEVICE)
  
  summary(vaeNet, input_size=preprocessed_final_shape, device=DEVICE_STR)

  optimizer = torch.optim.Adam(vaeNet.parameters(), lr=0.001)

  game = vzd.DoomGame()
  game.set_doom_game_path("../../../build/bin/doom.wad")
  game.set_doom_scenario_path("../../../build/bin/doom.wad")
  game.set_doom_map(args.map)
  game.set_mode(vzd.Mode.PLAYER)
  game.set_episode_start_time(10)
  game.set_episode_timeout(999999999)
  #game.add_game_args("+freelook 1")

  # Use other config file if you wish.
  #game.load_config(args.config)
  game.set_render_hud(True)
  game.set_render_minimal_hud(False)
  game.set_render_decals(True)
  game.set_render_particles(True)
  game.set_render_effects_sprites(True)
  game.set_render_corpses(True)
  #game.set_render_messages(False)
  
  game.set_screen_resolution(vzd.ScreenResolution.RES_320X256)

  # Set cv2 friendly format.
  game.set_screen_format(vzd.ScreenFormat.RGB24)

  # Enables labeling of the in game objects.
  game.set_labels_buffer_enabled(True)

  game.clear_available_game_variables()
  game.add_available_game_variable(vzd.GameVariable.POSITION_X)
  game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
  game.add_available_game_variable(vzd.GameVariable.POSITION_Z)

  actions = [
    [True,False,False,False,False,False,False,False],
    [False,True,False,False,False,False,False,False],
    [False,False,True,False,False,False,False,False],
    [False,False,False,True,False,False,False,False],
    [False,False,False,False,True,False,False,False],
    [False,False,False,False,False,True,False,False],
    [False,False,False,False,False,False,True,False],
    [False,False,False,False,False,False,False,True],
  ]
  game.set_available_buttons([
    vzd.Button.MOVE_LEFT,vzd.Button.MOVE_RIGHT,vzd.Button.TURN_LEFT,vzd.Button.TURN_RIGHT,
    vzd.Button.ATTACK,vzd.Button.MOVE_FORWARD,vzd.Button.MOVE_BACKWARD,vzd.Button.USE
  ])
  game.set_window_visible(True)
  game.init()
  episodes = 10000

  # Sleep time between actions in ms
  sleep_time = 0#1.0 / vzd.DEFAULT_TICRATE
  count = 0
  batch_size = 32
  lrDecayCount = 0
  label_map = {}
  
  batch_num = 0
  input_batch = []
  last_loss = inf
  
  for i in range(episodes):
    print("Episode #" + str(i + 1))
    seen_in_this_episode = set()

    # Not needed for the first episode but the loop is nicer.
    game.new_episode()
    
    while not game.is_episode_finished():
      # Get the state
      state = game.get_state()
      screen_tensor = preprocess_screenbuffer(state.screen_buffer)
      input_batch.append(screen_tensor)

      count += 1
      if count % batch_size == 0:
        
        inputs = torch.stack(input_batch).to(DEVICE)
        reconst, input, mu, logvar = vaeNet(inputs)
        loss = vaeNet.loss_function(reconst, inputs, mu, logvar)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
        input_batch = []
        batch_num += 1
        last_loss = loss.item()
        
        print(f"[Step: {count}, Batch: {batch_num}, Episode: {i}] Loss: {loss:.10f}, lr: {optimizer.param_groups[0]['lr']:.10f}")
            
        if batch_num % 500 == 0:
          torch.save(vaeNet.state_dict(), modelFilepath)
          print("Model saved to " + modelFilepath)

      if game.get_mode() == vzd.Mode.SPECTATOR:
        game.advance_action()
      else:
        if last_loss < 1:
          game.make_action(choice(actions), 4)
          lrDecayCount = 0

      new_lr = min(0.001, max(0.000001, 0.95 ** ((4000+lrDecayCount) // batch_size)))
      optimizer.param_groups[0]['lr'] = new_lr
      if new_lr <= 0.000002:
        lrDecayCount = 0
      else:
        lrDecayCount += 1

    print("Episode finished!")
    print("=====================")

  #cv2.destroyAllWindows()
  game.close()
