
import os
import typing
import math

import torch
import torchvision.transforms.functional as torchvisfunc
from torchsummary import summary
import numpy as np

from PyQt6.QtCore import pyqtSlot, pyqtSignal, QObject, QThread


import vizdoom as vzd

from doom_env import DoomEnv
from doom_vae import DoomVAE

LATENT_SPACE_SIZE = 1024
PREPROCESS_RES_H_W = (128,160) # Must be (H,W)!
PREPROCESS_FINAL_SHAPE_C_H_W = (3, PREPROCESS_RES_H_W[0], PREPROCESS_RES_H_W[1])

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)


class DoomAgentOptions:
  num_episodes=10000 
  batch_size = 8
  model_filepath     = "./checkpoints/doom_vae.model"
  optimizer_filepath = "./checkpoints/doom_vae.optim"
  doom_map = "E1M1"

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
  @pyqtSlot()
  def request_screenbuf(self):
     self.screenbuf_signal.emit(self._last_screenbuf.detach().clone())
  @pyqtSlot()
  def stop(self):
    self._force_killed = True
  
  
  def __init__(self, options:DoomAgentOptions=DoomAgentOptions(), parent: typing.Optional[QObject] = None) -> None:
    super().__init__(parent)
    
    self.options = options
    self._force_killed = False
    self.doom_env = None
    self._last_screenbuf = torch.Tensor()
    self.lr_min = 0.0002
    self.lr_max = 0.0002
    self.batches_per_action = 1
    
  def __del__(self):
    self._force_killed = True
    self.wait()
    self.cleanup()
 
 
  def cleanup(self):
    if self.doom_env != None:
      self.doom_env.game.close()

  def build_vae_network(self):
    net = DoomVAE(PREPROCESS_FINAL_SHAPE_C_H_W, LATENT_SPACE_SIZE)
    model_filepath = self.options.model_filepath
    if os.path.exists(model_filepath):
      net.load_state_dict(torch.load(model_filepath))
      print("Model loaded from " + model_filepath)
    else:
      print("Could not find file " + model_filepath)

    return net.to(DEVICE)

  def preprocess_screenbuffer(screenbuf):
    screenbuf = torchvisfunc.to_tensor(screenbuf)
    screenbuf = torchvisfunc.resize(screenbuf, PREPROCESS_RES_H_W)
    #screenbuf = torchvisfunc.normalize(screenbuf, (0.485,0.456,0.406), (0.229,0.224,0.225))
    #screenbuf = screenbuf[:,0:non_hud_height,:]
    assert screenbuf.shape == PREPROCESS_FINAL_SHAPE_C_H_W
    assert np.count_nonzero(np.isnan(screenbuf.numpy())) == 0
    return screenbuf
  
  def deprocess_screenbuffer(screenbuf_tensor):
    #screenbuf = torchvisfunc.normalize(screenbuf_tensor, (0.,0.,0.), (1.0/0.229,1.0/0.224,1.0/0.225))
    #screenbuf = torchvisfunc.normalize(screenbuf, (-0.485,-0.456,-0.406), (1.,1.,1.))
    return screenbuf_tensor
  
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
    
    print("Model Summary...")
    summary(self.vae_net, input_size=PREPROCESS_FINAL_SHAPE_C_H_W, device=DEVICE_STR)
    
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
        
        screen_tensor = DoomAgentThread.preprocess_screenbuffer(state.screen_buffer)
        if np.count_nonzero(screen_tensor.numpy()) == 0:
          print("************ ALL ZEROES!!!!!!! ************")
          exit()
          
        input_batch.append(screen_tensor)
        self._last_screenbuf = screen_tensor
        if epIdx == 0: self.screenbuf_available_signal.emit()
        
        count += 1
        if count % batch_size == 0:
          
          inputs = torch.stack(input_batch).to(DEVICE)
          reconst, input, mu, logvar = self.vae_net(inputs)
          
          kl_beta = math.sin(0.5*math.pi * float(kl_decay_count) / KL_CYCLE_LENGTH)
          
          if kl_decay_count < KL_CYCLE_LENGTH: kl_decay_count += 1
          else: 
            kl_wait_count += 1
            if kl_wait_count > KL_CYCLE_LENGTH:
              kl_wait_count = 0
              kl_decay_count = 0
          
          loss, reconst_loss, kld_loss = self.vae_net.loss_function(reconst, inputs, mu, logvar, kl_beta)
          
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
          
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
        
          input_batch = []
          batch_num += 1
          batches_since_last_action += 1
          last_loss = loss.item()
          
          if count % 10 == 0:
            print(f"[Batch #: {batch_num}, Episode #: {epNum}] Loss: {loss:.5f}, Reconst Loss: {reconst_loss:.3f}, KLD Loss: {kld_loss:.3f}, kl_beta: {kl_beta:.5f}, lr: {self.optimizer.param_groups[0]['lr']:.6f}")
              
          if batch_num % 1000 == 0:
            torch.save(self.vae_net.state_dict(), model_filepath)
            print("Model saved to " + model_filepath)
            torch.save(self.optimizer.state_dict(), optim_filepath)
            print("Optimizer saved to " + optim_filepath)
            self.vae_saved_signal.emit()

        if self.doom_env.game.get_mode() == vzd.Mode.SPECTATOR:
          self.doom_env.game.advance_action()

        if batches_since_last_action >= self.batches_per_action:
          #if last_loss < 200:
          next_state, reward, done = self.doom_env.step(self.doom_env.random_action())
          batches_since_last_action = 0
        
        #if next_state == None:
        #  done = True
        #  continue

        
        #TODO: state = next_state

