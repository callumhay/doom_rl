import torch
import numpy as np

from sd_encode_decode import SDEncoder, SDDecoder
from doom_rssm import DoomRSSM
from doom_models import DoomDiscreteActionModel
from config import CONTINUOUS_NET_TYPE, DISCRETE_NET_TYPE

class DoomEvaluator(object):
  
  def __init__(self, config, device) -> None:
    self.config = config
    self.device = device
    
  def load_model(self, config, saved_dict):
    observation_shape = config.observation_shape
    action_size = config.action_size
    deter_size  = config.rssm_info['deter_size']
    if config.rssm_info['type'] == CONTINUOUS_NET_TYPE:
      stoch_size = config.rssm_info['stoch_size']
    elif config.rssm_info['type'] == DISCRETE_NET_TYPE:
      category_size = config.rssm_info['category_size']
      class_size = config.rssm_info['class_size']
      stoch_size = category_size * class_size
      
    obs_c, obs_h, obs_w = observation_shape
    modelstate_size = stoch_size + deter_size
    
    self.observation_encoder = SDEncoder(
      **config.encoder_decoder_config, 
      res_h=obs_h, res_w=obs_w, in_channels=obs_c
    ).to(self.device).eval()
    
    self.observation_decoder = SDDecoder(
      **config.encoder_decoder_config,
      res_h=obs_h, res_w=obs_w, in_channels=obs_c, out_ch=obs_c, 
      modelstate_size=modelstate_size
    ).to(self.device).eval()
    
    embedding_size = self.observation_decoder.embedding_size
    full_embed_size = embedding_size+4
    
    self.rssm = DoomRSSM(action_size, full_embed_size, self.device, config.rssm_info).to(self.device).eval()
    self.action_model = DoomDiscreteActionModel(action_size, deter_size, stoch_size, config.actor_info, config.epsilon_info).to(self.device).eval()
    if saved_dict != None:
      self.rssm.load_state_dict(saved_dict['rssm'])
      self.observation_encoder.load_state_dict(saved_dict['observation_encoder'])
      self.observation_decoder.load_state_dict(saved_dict['observation_decoder'])
      self.action_model.load_state_dict(saved_dict['action_model'])
    
  def eval_saved_agent(self, env, saved_dict, num_episodes=100):
    self.load_model(self.config, saved_dict)
    eval_scores = []
    for episode in range(num_episodes):
      score = 0
      observation, done = env.reset(), False
      position = env.player_pos()
      heading = env.player_heading()
      prev_state = self.rssm.init_state(1)
      prev_action = torch.zeros(1, self.config.action_size).to(self.device)
      
      while not done:
        with torch.no_grad():
          observation_embed = torch.cat((
              torch.flatten(self.observation_encoder(torch.tensor(observation, device=self.device).unsqueeze(0)), 1),
              torch.tensor(position, device=self.device).unsqueeze(0), 
              torch.tensor([heading], device=self.device).unsqueeze(0)
          ), dim=-1).to(dtype=torch.float32)
          _, posterior_state = self.rssm.observe(observation_embed, prev_action, not done, prev_state)
          modelstate = self.rssm.get_model_state(posterior_state)
          action, _  = self.action_model(modelstate)
          prev_state = posterior_state
          prev_action = action
          
        next_observation, reward, done = env.step(action.squeeze(0).cpu().numpy())
        next_position = env.player_pos()
        next_heading  = env.player_heading()
        score += reward
        observation = next_observation
        position = next_position
        heading  = next_heading
      
      eval_scores.append(score)
      print(f"Episode {episode} Complete, Score = {score}")
      
    mean_score = np.mean(eval_scores)
    print("Average evaluation score for model at " + model_path + " = " + str(mean_score))
    return mean_score
  