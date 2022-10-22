
import os
import typing
import math
import time

import yaml
from yaml.loader import SafeLoader

import torch
import torchvision.transforms.functional as torchvisfunc
from torchsummary import summary
import numpy as np

from PyQt6.QtCore import pyqtSlot, pyqtSignal, QObject, QThread
import vizdoom as vzd
from doom_env import DoomEnv, PREPROCESS_FINAL_SHAPE_C_H_W
from sd_vae import SDVAE

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


class DoomAgentOptions:
  num_episodes = 10000 
  batch_size = 32
  model_filepath     = "./checkpoints/doom_vae.model"
  optimizer_filepath = "./checkpoints/doom_vae.optim"
  doom_map = "E1M2"

class DoomAgentThread(QThread):
  # Signals
  screenbuf_signal = pyqtSignal(torch.Tensor, name="screenbuf")
  screenbuf_available_signal = pyqtSignal(name="screenbuf_available")
  vae_saved_signal = pyqtSignal(name="vae_saved")
  
  # Slots
  @pyqtSlot(float,float)
  def set_lr_min_max(self, lr_min, lr_max):
    self.lr_min = lr_min
    self.lr_max = lr_max
    print(f"Updating learning rate range to [{lr_min},{lr_max}]")
  @pyqtSlot(int)
  def set_batches_per_action(self, bpa):
    self.batches_per_action = bpa
    print(f"Updating batches per action to {bpa}")
  @pyqtSlot(bool)
  def toggle_play(self, toggle):
    self.is_play_enabled = toggle
  @pyqtSlot()
  def request_screenbuf(self):
     self.screenbuf_signal.emit(self._last_screenbuf.detach().clone())
  @pyqtSlot(int)
  def set_fps(self, fps):
    self.screenbuf_signal_fps = fps
  @pyqtSlot()
  def stop(self):
    self._force_killed = True
  
  
  def __init__(self, options:DoomAgentOptions=DoomAgentOptions(), parent: typing.Optional[QObject] = None) -> None:
    super().__init__(parent)
    
    self.options = options
    self._force_killed = False
    self.doom_env = None
    self._last_screenbuf = torch.Tensor()
    self.lr_min = 0.00001
    self.lr_max = 0.00001
    self.batches_per_action = 1
    self.is_play_enabled = True
    self.screenbuf_signal_fps = 8
    
    self.vae_config_info = None
    with open("config/sd_vae_config.yaml", "r") as ymlfile:
      self.vae_config_info = yaml.load(ymlfile, Loader=SafeLoader)
    
  def __del__(self):
    self._force_killed = True
    self.wait()
    self.cleanup()
 
 
  def cleanup(self):
    if self.doom_env != None:
      self.doom_env.game.close()

  def build_vae_network(self):
    model_filepath = self.options.model_filepath
    model_info = None
    if os.path.exists(model_filepath):
      model_dict = torch.load(model_filepath)
      self.vae_config_info = model_dict['config']
      model_info  = model_dict['model']
    else:
      print("Could not find file " + model_filepath)
  
    net = SDVAE(self.vae_config_info['sd_vae_config'], self.vae_config_info['embed_dim'])
    if model_info != None:
      net.load_state_dict(model_info)
      print("Model loaded from " + model_filepath)
    
    '''
    net = DoomVAE(PREPROCESS_FINAL_SHAPE_C_H_W)
    model_filepath = self.options.model_filepath
    if os.path.exists(model_filepath):
      net.load_state_dict(torch.load(model_filepath))
      print("Model loaded from " + model_filepath)
    else:
      print("Could not find file " + model_filepath)
    '''
    return net.to(DEVICE)
  
  def run(self):
    self._force_killed = False
    model_filepath = self.options.model_filepath
    optim_filepath = self.options.optimizer_filepath
    doom_map = self.options.doom_map
    
    # Setup our model and related variables
    self.vae_net = self.build_vae_network()
    self.vae_net.train()
    self.optimizer = torch.optim.Adam(self.vae_net.parameters(), lr=self.lr_max)
    if os.path.exists(optim_filepath):
      self.optimizer.load_state_dict(torch.load(optim_filepath))
      print("Optimizer loaded from " + optim_filepath)
    
    #print("Model Summary...")
    #summary(self.vae_net, input_size=PREPROCESS_FINAL_SHAPE_C_H_W, device=DEVICE_STR)
    
    # Setup our environment (for simulation)
    self.doom_env = DoomEnv(doom_map)
    
    count = 0
    kl_decay_count = 0
    kl_wait_count = 0
    KL_CYCLE_LENGTH = 10000.0
    lrDecayCount = 0
    batch_num = 0
    batch_size = self.options.batch_size
    model_filepath = self.options.model_filepath
    input_batch = []
    last_loss = 999999
    batches_since_last_action = 0
    loss_correction_attempts = 0
    last_time_ns = time.perf_counter_ns()
    
    # Run the simulation across the set number of episodes
    for epIdx in range(self.options.num_episodes):
      if self._force_killed:
        print("Doom Agent Thread has been killed, terminating.")
        self.cleanup()
        break
      
      epNum = epIdx+1
      print(f"Episode #{epNum}")
      
      state = self.doom_env.reset()
      
      while not self.doom_env.game.is_episode_finished():

        new_lr = self.lr_min + (self.lr_max-self.lr_min)*math.cos(math.pi*lrDecayCount/(500*batch_size))
        self.optimizer.param_groups[0]['lr'] = new_lr
        if new_lr <= self.lr_min:
          lrDecayCount = 0
        else:
          lrDecayCount += 1
          

         # TODO: Fix this once we're doing actual RL
        state = self.doom_env.game.get_state() 
        
        if self._force_killed:
          break
        
        screen_tensor = DoomEnv.preprocess_screenbuffer(state.screen_buffer)
        if np.count_nonzero(screen_tensor.numpy()) == 0:
          print("************ ALL ZEROES!!!!!!! ************")
          exit()
          
        input_batch.append(screen_tensor)
        self._last_screenbuf = screen_tensor
        if epIdx == 0:
          self.screenbuf_available_signal.emit()
          
        curr_time_ns = time.perf_counter_ns()
        delta_time_ms = (curr_time_ns - last_time_ns) // 1e6
        if self.screenbuf_signal_fps != 0 and delta_time_ms >= 1000*(1.0 / float(self.screenbuf_signal_fps)):
          self.request_screenbuf()
          last_time_ns = curr_time_ns
        
        count += 1
        if len(input_batch) >= batch_size:

          
          #kl_beta = math.sin(0.5*math.pi * float(kl_decay_count) / KL_CYCLE_LENGTH)
    
          if kl_decay_count < KL_CYCLE_LENGTH: kl_decay_count += 1
          else: 
            kl_wait_count += 1
            if kl_wait_count > KL_CYCLE_LENGTH:
              kl_wait_count = 0
              kl_decay_count = 0
          
          # Train the agent...
          self.optimizer.zero_grad()
          inputs = torch.stack(input_batch).to(DEVICE)
          
          loss, reconst_loss, kld_loss = self.vae_net.training_step(inputs)
          
          #obs_reconst, loss, reconst_loss, kld_loss = self.vae_net(inputs, device=DEVICE)

          if math.isnan(loss) or math.isinf(loss):
            print("Loss is inf/nan, exiting")
            self._force_killed = True
            break
          elif loss > 100000000:
            print("Loss is exploding, attempting to course correct...")
            batches_since_last_action = self.batches_per_action
            self.batches_per_action *= 2
            self.lr_min *= 0.1
            self.lr_max *= 0.1
            loss_correction_attempts += 1
          else:
            loss_correction_attempts = 0
          
          if loss_correction_attempts > 5:
            print("Cannont course correct, loss is exploding. Exiting.")
            self._force_killed = True
            break
          
          loss.backward()
          self.optimizer.step()
        
          input_batch = []
          batch_num += 1
          batches_since_last_action += 1
          last_loss = loss.item()
          
          if count % 10 == 0:
            print(f"[Batch #: {batch_num}, Episode #: {epNum}] Loss: {loss:.5f}, Reconst Loss: {reconst_loss:.5f}, KLD Loss: {kld_loss:.5f}, lr: {self.optimizer.param_groups[0]['lr']:.7f}")
              
          if batch_num % 500 == 0:
            #torch.save(self.vae_net.state_dict(), model_filepath)
            torch.save({
              'config': self.vae_config_info,
              'model' : self.vae_net.state_dict(),
            }, model_filepath)
            
            print("Model saved to " + model_filepath)
            torch.save(self.optimizer.state_dict(), optim_filepath)
            print("Optimizer saved to " + optim_filepath)
            self.vae_saved_signal.emit()

        if self.doom_env.game.get_mode() == vzd.Mode.SPECTATOR:
          self.doom_env.game.advance_action()

        if self.is_play_enabled and batches_since_last_action >= self.batches_per_action:
          next_state, reward, done = self.doom_env.step(self.doom_env.random_action())
          batches_since_last_action = 0


