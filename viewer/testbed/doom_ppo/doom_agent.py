import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np

from doom_env import NUM_LABEL_CLASSES


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
  nn.init.orthogonal_(layer.weight, std)
  nn.init.constant_(layer.bias, bias_const)
  return layer

class DoomAgent(nn.Module):
  def __init__(self, envs, args):
    super(DoomAgent, self).__init__()
    
    net_output_size  = args.net_output_size  # Original was 512
    lstm_hidden_size = args.lstm_hidden_size # Original was 128
    
    out_channel_list = args.conv_channels # Original was [32, 64, 64]
    kernel_size_list = args.conv_kernels  # Original was [ 8,  4,  3]
    stride_list      = args.conv_strides  # Original was [ 4,  2,  1]
    padding_list     = args.conv_paddings # Original was [ 0,  0,  0]
    
    self.network = nn.Sequential()
    curr_channels, curr_height, curr_width = envs.observation_shape()

    for out_channels, kernel_size, stride, padding in zip(out_channel_list, kernel_size_list, stride_list, padding_list):
      self.network.append(layer_init(nn.Conv2d(
        in_channels=curr_channels, out_channels=out_channels, 
        kernel_size=kernel_size, stride=stride, padding=padding
      )))
      self.network.append(nn.ReLU())
      
      curr_width  = int((curr_width-kernel_size + 2*padding) / stride + 1)
      curr_height = int((curr_height-kernel_size + 2*padding) / stride + 1)
      curr_channels = out_channels
    self.network.append(nn.Flatten())
    
    conv_output_size = curr_width*curr_height*curr_channels
    if net_output_size is None:
      net_output_size = conv_output_size
    else:
      #self.network.append(layer_init(nn.Linear(conv_output_size, net_output_size))) 
      #self.network.append(nn.ReLU())
      self.network.append(layer_init(nn.Linear(conv_output_size, net_output_size), std=nn.init.calculate_gain('leaky_relu', 0.2))) 
      self.network.append(nn.LeakyReLU(0.2))
    
    self.lstm = nn.LSTM(net_output_size, lstm_hidden_size)
    for name, param in self.lstm.named_parameters():
      if "bias" in name: nn.init.constant_(param, 0)
      elif "weight" in name: nn.init.orthogonal_(param, 1.0)
    
    # Actor network init
    curr_layer_input_size = lstm_hidden_size
    self.actor = nn.Sequential()
    for _ in range(args.actor_critic_num_hidden):
      self.actor.append(layer_init(nn.Linear(curr_layer_input_size, args.actor_critic_hidden_size), std=0.01))
      self.actor.append(nn.LeakyReLU())
      curr_layer_input_size = args.actor_critic_hidden_size
    self.actor.append(layer_init(nn.Linear(curr_layer_input_size, envs.action_space_size()), std=0.01))
    
    # Critic network init
    curr_layer_input_size = lstm_hidden_size
    self.critic = nn.Sequential()
    for _ in range(args.actor_critic_num_hidden):
      self.critic.append(layer_init(nn.Linear(curr_layer_input_size, args.actor_critic_hidden_size), std=1))
      self.critic.append(nn.LeakyReLU())
      curr_layer_input_size = args.actor_critic_hidden_size
    self.critic.append(layer_init(nn.Linear(curr_layer_input_size, 1), std=1))
    
    # Classifier network init
    if args.use_classifier:
      curr_layer_input_size = net_output_size
      self.classifier = nn.Sequential()
      for _ in range(args.classifier_num_hidden):
        self.classifier.append(layer_init(nn.Linear(curr_layer_input_size, args.classifier_hidden_size), std=0.1))
        self.classifier.append(nn.ELU())
        self.classifier.append(nn.Dropout(0.2))
        curr_layer_input_size = args.classifier_hidden_size 
      self.classifier.append(layer_init(nn.Linear(curr_layer_input_size, NUM_LABEL_CLASSES), std=1))
    else:
      self.classifier = None

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

    if self.classifier is not None:
      classifier_logits = self.classifier(conv_output)
      classifier_logits = classifier_logits.view(-1, classifier_logits.shape[-1])
    else:
      classifier_logits = None
    return action, actor_dist.log_prob(action), actor_dist.entropy(), self.critic(new_hidden), lstm_state, classifier_logits

      