import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np

from doom_env import PREPROCESS_FINAL_SHAPE_C_H_W, NUM_LABEL_CLASSES

'''
# Original implementation of the network:
self.network = nn.Sequential(
  layer_init(nn.Conv2d(1, 32, 8, stride=4)),
  nn.ReLU(),
  layer_init(nn.Conv2d(32, 64, 4, stride=2)),
  nn.ReLU(),
  layer_init(nn.Conv2d(64, 64, 3, stride=1)),
  nn.ReLU(),
  nn.Flatten(),
  layer_init(nn.Linear(64 * 7 * 7, 512)),
  nn.ReLU(),
)
'''

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, bias_const)
  return layer

class DoomAgent(nn.Module):
  def __init__(self, action_space_size, args):
    super(DoomAgent, self).__init__()
    
    net_output_size  = args.net_output_size # Original was 512
    lstm_hidden_size = args.lstm_hidden_size     # Original was 128
    
    out_channel_list = args.conv_channels # Original was [32, 64, 64]
    kernel_size_list = args.conv_kernels  # Original was [ 8,  4,  3]
    stride_list      = args.conv_strides  # Original was [ 4,  2,  1]
    padding_list     = args.conv_paddings # Original was [ 0,  0,  0]
    
    self.network = nn.Sequential()
    curr_channels, curr_height, curr_width = PREPROCESS_FINAL_SHAPE_C_H_W

    for out_channels, kernel_size, stride, padding in zip(out_channel_list, kernel_size_list, stride_list, padding_list):
      self.network.append(nn.Conv2d(
        in_channels=curr_channels, out_channels=out_channels, 
        kernel_size=kernel_size, stride=stride, padding=padding
      ))
      self.network.append(nn.ReLU())
      
      curr_width  = int((curr_width-kernel_size + 2*padding) / stride + 1)
      curr_height = int((curr_height-kernel_size + 2*padding) / stride + 1)
      curr_channels = out_channels

    conv_output_size = curr_width*curr_height*curr_channels
    
    self.network.append(nn.Flatten())
    self.network.append(layer_init(nn.Linear(conv_output_size, net_output_size)))
    
    self.lstm = nn.LSTM(net_output_size, lstm_hidden_size)
    for name, param in self.lstm.named_parameters():
      if "bias" in name: nn.init.constant_(param, 0)
      elif "weight" in name: nn.init.orthogonal_(param, 1.0)
        
    self.actor      = layer_init(nn.Linear(lstm_hidden_size, action_space_size), std=0.01)
    self.critic     = layer_init(nn.Linear(lstm_hidden_size, 1), std=1)
    
    # Classifier network init
    curr_layer_input_size = net_output_size if args.classifier_num_hidden == 0 else args.classifier_hidden_size
    self.classifier = nn.Sequential()
    for _ in range(args.classifier_num_hidden):
      self.classifier.append(layer_init(nn.Linear(curr_layer_input_size, args.classifier_hidden_size)))
      self.classifier.append(nn.Tanh())
      curr_layer_input_size = args.classifier_hidden_size 
    self.classifier.append(layer_init(nn.Linear(curr_layer_input_size, NUM_LABEL_CLASSES)))
    

  def get_states(self, x, lstm_state, done):
    hidden = self.network(x)

    # LSTM logic
    batch_size = lstm_state[0].shape[1]
    hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
    done = done.reshape((-1, batch_size))
    new_hidden = []
    for h, d in zip(hidden, done):
      h, lstm_state = self.lstm(
        h.unsqueeze(0),
        ((1.0 - d).view(1, -1, 1) * lstm_state[0], (1.0 - d).view(1, -1, 1) * lstm_state[1],),
      )
      new_hidden += [h]
    new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
    return hidden, new_hidden, lstm_state

  def get_value(self, x, lstm_state, done):
    _, new_hidden, _ = self.get_states(x, lstm_state, done)
    return self.critic(new_hidden)

  def get_action_and_value(self, x, lstm_state, done, action=None):
    _, new_hidden, lstm_state = self.get_states(x, lstm_state, done)
    
    actor_logits = self.actor(new_hidden)
    probs = td.Categorical(logits=actor_logits)
    if action is None:
      action = probs.sample()

    return action, probs.log_prob(action), probs.entropy(), self.critic(new_hidden), lstm_state
  

  def get_action_value_classify(self, x, lstm_state, done, action):
    conv_output, new_hidden, lstm_state = self.get_states(x, lstm_state, done)
    
    actor_logits = self.actor(new_hidden)
    actor_dist = td.Categorical(logits=actor_logits)
    if action is None:
      action = actor_dist.sample()

    classifier_logits = self.classifier(conv_output)
    classifier_logits = classifier_logits.view(-1, classifier_logits.shape[-1])
    return action, actor_dist.log_prob(action), actor_dist.entropy(), self.critic(new_hidden), lstm_state, classifier_logits

      