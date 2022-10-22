import vizdoom as vzd

import torch
import torchvision.transforms.functional as torchvisfunc
import torchshow as ts
import argparse
from random import choice
import numpy as np
from torchsummary import summary

from doom_classifier import DoomClassifier

DEVICE_STR = 'cpu'
DEVICE = torch.device(DEVICE_STR)

preprocess_res = (128,160) # Must be (H,W)!
#non_hud_height = 83
preprocessed_final_shape = (3, preprocess_res[0], preprocess_res[1])#(3, non_hud_height, preprocess_res[1])
def preprocess_screenbuffer(screenbuf: np.ndarray) -> torch.Tensor:
  screenbuf = torchvisfunc.to_tensor(screenbuf)
  screenbuf = torchvisfunc.resize(screenbuf, preprocess_res)
  screenbuf = torchvisfunc.normalize(screenbuf, (0.485,0.456,0.406), (0.229,0.224,0.225))
  #screenbuf = screenbuf[:,0:non_hud_height,:]
  #ts.show(screenbuf)

  screenbuf.unsqueeze_(0)
  assert screenbuf.shape == (1,preprocessed_final_shape[0],preprocessed_final_shape[1],preprocessed_final_shape[2])
  return screenbuf


if __name__ =="__main__":
  parser = argparse.ArgumentParser(description="Train a Doom object classifier network.")
  parser.add_argument('-net', '--net', '--n', '-n', help="An existing classifier network to load.", default="")
  args = parser.parse_args()
  
  if torch.cuda.is_available():
    DEVICE_STR = 'cuda'
    DEVICE = torch.device(DEVICE_STR)
    #torch.backends.cudnn.benchmark = True   

  classifierFilepath = "./checkpoints/doom_classifier.model"
  num_classes = 256
  classifier = DoomClassifier(preprocessed_final_shape, num_classes=num_classes, hidden_sizes=[1024,512]).train()
  if (len(args.net) > 0):
    classifier.load_state_dict(torch.load(classifierFilepath))
  classifier = classifier.to(DEVICE)
  
  summary(classifier, input_size=preprocessed_final_shape, device=DEVICE_STR)

  optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
  loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum').to(DEVICE)
  
  game = vzd.DoomGame()
  game.set_doom_game_path("../../build/bin/doom.wad")
  game.set_doom_scenario_path("../../build/bin/doom.wad")
  game.set_doom_map("E1M1")
  game.set_mode(vzd.Mode.ASYNC_PLAYER)
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
  count = 1
  lrDecayCount = 0
  label_map = {}
  
  for i in range(episodes):
    print("Episode #" + str(i + 1))
    seen_in_this_episode = set()

    # Not needed for the first episode but the loop is nicer.
    game.new_episode()
    
    while not game.is_episode_finished():
      # Get the state
      state = game.get_state()
      screen_tensor = preprocess_screenbuffer(state.screen_buffer)
      screen_tensor = screen_tensor.to(DEVICE)
      
      # Attempt classification...
      actClasses = classifier(screen_tensor).to(DEVICE)
      # Get target labels (classes) that are currently visible
      tgtClasses = torch.zeros(1, num_classes)
      for l in state.labels: tgtClasses[0][l.value] = 1.0
      tgtClasses = tgtClasses.to(DEVICE)
      
      # Calculate the cross-entropy loss between what was classified and what labels
      # are actually present in the current screen buffer
      loss = loss_fn(actClasses, tgtClasses).to(DEVICE)
      
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      
      for l in state.labels: label_map[l.value] = l.object_name 
      
      if count % vzd.DEFAULT_TICRATE == 0:
        print(f"[Step: {count}, Episode: {i}] Loss: {loss:.10f}, lr: {optimizer.param_groups[0]['lr']:.10f}")
        
        targets = []
        identified = []
        idx = 0
        for a,t in zip(actClasses[0], tgtClasses[0]):
          net_val = a.item()
          target_val = t.item()
          if target_val == 1:
            targets.append(label_map[idx])
          if net_val > 0.8:
            identified.append(label_map[idx] if idx in label_map else "???")
          idx += 1
        print("Targets:    " + ', '.join(targets))
        print("Identified: " + ', '.join(identified))
           
      if count % 5000 == 0:
        torch.save(classifier.state_dict(), classifierFilepath)
        print("Model saved to " + classifierFilepath)

      if loss.item() < 0.5:
        lrDecayCount = 0
        game.make_action(choice(actions))
        game.advance_action()
      else:
        new_lr = max(0.00000001, 0.95 ** ((3000+lrDecayCount) // 70))
        optimizer.param_groups[0]['lr'] = new_lr
        if new_lr <= 0.00000002:
          lrDecayCount = 0
        else:
          lrDecayCount += 1

      # Get labels buffer, that is always in 8-bit grey channel format.
      # Show only visible game objects (enemies, pickups, exploding barrels etc.), each with a unique label.
      # Additional labels data are available in state.labels.
      
      #if labels is not None:
      #    cv2.imshow('ViZDoom Labels Buffer', color_labels(labels))

      #for l in state.labels:
      #    if l.object_name in ["Medkit", "GreenArmor"]:
      #        draw_bounding_box(screen, l.x, l.y, l.width, l.height, doom_blue_color)
      #    else:
      #        draw_bounding_box(screen, l.x, l.y, l.width, l.height, doom_red_color)
      #cv2.imshow('ViZDoom Screen Buffer', screen)
      #cv2.waitKey(sleep_time)

      # Make random action
      #game.make_action(choice(actions))

      #print("State #" + str(state.number))
      #print("Player position: x:", state.game_variables[0], ", y:", state.game_variables[1], ", z:", state.game_variables[2])
      #print("Labels:")

      # Print information about objects visible on the screen.
      # object_id identifies a specific in-game object.
      # It's unique for each object instance (two objects of the same type on the screen will have two different ids).
      # object_name contains the name of the object (can be understood as type of the object).
      # value tells which value represents the object in labels_buffer.
      # Values decrease with the distance from the player.
      # Objects with higher values (closer ones) can obscure objects with lower values (further ones).
      '''
      if count % 35 == 0:
        labels = state.labels_buffer
        for l in state.labels:
          seen_in_this_episode.add(l.object_name)
          #print("---------------------")
          print("Label:", l.value, ", object id:", l.object_id, ", object name:", l.object_name)
          #print("Object position: x:", l.object_position_x, ", y:", l.object_position_y, ", z:", l.object_position_z)
          # Other available fields (position and velocity and bounding box):
          #print("Object rotation angle", l.object_angle, "pitch:", l.object_pitch, "roll:", l.object_roll)
          #print("Object velocity x:", l.object_velocity_x, "y:", l.object_velocity_y, "z:", l.object_velocity_z)
          #print("Bounding box: x:", l.x, ", y:", l.y, ", width:", l.width, ", height:", l.height)
      '''
      count +=1
      

    print("Episode finished!")
    print("=====================")
    print("Unique objects types seen in this episode:")
    for l in seen_in_this_episode:
      print(l)
    print("************************")

  #cv2.destroyAllWindows()
  game.close()