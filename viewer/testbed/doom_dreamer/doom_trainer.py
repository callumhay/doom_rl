
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as td

import numpy as np

from diagonal_gaussian_distribution import DiagonalGaussianDistribution
from replay_memory import ReplayMemory
from doom_rssm import DoomRSSM, RSSMState
from doom_models import DoomDenseModel, DoomDiscreteActionModel
from sd_encode_decode import SDEncoder, SDDecoder
from sd_reconstruction_loss import SDReconstructionLoss

from config import Config, CONTINUOUS_NET_TYPE, DISCRETE_NET_TYPE
from util import FreezeParameters, get_parameters

class DoomTrainer:
  def __init__(self, config: Config, device: torch.device) -> None:
    self.device = device
    self.config = config
    self._model_init(config)
    self._optimizer_init(config)
    
  def _model_init(self, config: Config):
    observation_shape = config.observation_shape
    action_size = config.action_size
    deter_size  = config.rssm_info['deter_size']
    if config.rssm_info['type'] == CONTINUOUS_NET_TYPE:
      stoch_size = config.rssm_info['stoch_size']
    elif config.rssm_info['type'] == DISCRETE_NET_TYPE:
      category_size = config.rssm_info['category_size']
      class_size = config.rssm_info['class_size']
      stoch_size = category_size * class_size
    
    self.replay_memory = ReplayMemory(config.capacity, observation_shape, action_size, config.seq_len, config.batch_size)
    
    obs_c, obs_h, obs_w = observation_shape
    modelstate_size = stoch_size + deter_size
    
    self.observation_encoder = SDEncoder(
      **config.encoder_decoder_config, 
      res_h=obs_h, res_w=obs_w, in_channels=obs_c
    ).to(self.device)
    
    self.observation_decoder = SDDecoder(
      **config.encoder_decoder_config,
      res_h=obs_h, res_w=obs_w, in_channels=obs_c, out_ch=obs_c, 
      modelstate_size=modelstate_size
    ).to(self.device)
    
    self.reconst_loss = SDReconstructionLoss().to(self.device)
    embedding_size = self.observation_decoder.embedding_size
    
    print(f"Stochastic size of    {stoch_size}")
    print(f"Deterministic size of {deter_size}")
    print(f"Modelstate size of    {modelstate_size}.")
    print(f"Embedding shape of    {self.observation_decoder.embedding_shape} => {embedding_size} size.")
    
    self.rssm = DoomRSSM(action_size, embedding_size, self.device, config.rssm_info).to(self.device)
    self.action_model = DoomDiscreteActionModel(action_size, deter_size, stoch_size, config.actor_info, config.epsilon_info).to(self.device)
    self.pos_ori_model = DoomDenseModel((4,), modelstate_size, config.pos_ori_info).to(self.device)
    self.reward_decoder = DoomDenseModel((1,), modelstate_size, config.reward_info).to(self.device)
    self.value_model = DoomDenseModel((1,), modelstate_size, config.critic_info).to(self.device)
    self.target_value_model = DoomDenseModel((1,), modelstate_size, config.critic_info).to(self.device)
    self.target_value_model.load_state_dict(self.value_model.state_dict()) # Load the same initial weights as the value model
    self.discount_model = DoomDenseModel((1,), modelstate_size, config.discount_info).to(self.device)     
    
  def _optimizer_init(self, config):
    model_lr = config.lr_info['model']
    actor_lr = config.lr_info['actor']
    value_lr = config.lr_info['critic']
    
    self.world_models = [
      self.observation_encoder, self.observation_decoder, 
      self.rssm, self.reward_decoder, self.discount_model, self.pos_ori_model
    ]
    self.actor_models = [self.action_model]
    self.value_models = [self.value_model]
    self.actor_critic_models = [self.action_model, self.value_model]
    
    self.world_model_optimizer = optim.Adam(get_parameters(self.world_models), model_lr)
    self.actor_model_optimizer = optim.Adam(get_parameters(self.actor_models), actor_lr)
    self.value_model_optimizer = optim.Adam(get_parameters(self.value_models), value_lr)
    
    
  def initial_exploration(self, env) -> None:
    observation, done = env.reset(), False
    for _ in range(self.config.explore_steps):
      action = env.random_action()
      next_observation, reward, done = env.step(action)
      pos = env.player_pos()
      heading = env.player_heading()
      if done:
        self.replay_memory.add(observation, action, pos, heading, reward, done)
        observation, done = env.reset(), False
      else:
        self.replay_memory.add(observation, action, pos, heading, reward, done)
        observation = next_observation
  
  def update_target(self):
    mix = self.config.slow_target_fraction if self.config.use_slow_target else 1
    for param, target_param in zip(self.value_model.parameters(), self.target_value_model.parameters()):
        target_param.data.copy_(mix * param.data + (1 - mix) * target_param.data)
  
  def save_model(self, train_steps:int, episode:int) -> None:
    save_dict = self.get_save_dict()
    save_dict["train_steps"] = train_steps
    save_dict["episode"] = episode
    save_dict["csv_filepath"] = self.config.csv_filepath
    save_path = self.config.model_savepath_str(train_steps)
    torch.save(save_dict, save_path)
  
  def get_save_dict(self) -> Dict:
    return {
      "rssm": self.rssm.state_dict(),
      "observation_encoder": self.observation_encoder.state_dict(),
      "observation_decoder": self.observation_decoder.state_dict(),
      "reward_decoder": self.reward_decoder.state_dict(),
      "action_model": self.action_model.state_dict(),
      "value_model": self.value_model.state_dict(),
      "discount_model": self.discount_model.state_dict(),
      "pos_ori_model": self.pos_ori_model.state_dict(),
      "reconst_loss": self.reconst_loss.state_dict(),
    }
    
  def _attempt_load_state_dict(model, saved_dict_data,  model_name):
    try:
      model.load_state_dict(saved_dict_data, strict=False)
    except RuntimeError as e:
      print("Could not load " + model_name + ":")
      print(e)
      return False
    return True
    
  def load_save_dict(self, saved_dict: Dict) -> int:
    DoomTrainer._attempt_load_state_dict(self.rssm, saved_dict["rssm"], "RSSM")
    DoomTrainer._attempt_load_state_dict(self.observation_encoder, saved_dict["observation_encoder"], "Observation Encoder")
    DoomTrainer._attempt_load_state_dict(self.observation_decoder, saved_dict["observation_decoder"], "Observation Decoder")
    DoomTrainer._attempt_load_state_dict(self.reconst_loss, saved_dict["reconst_loss"], "Loss Model")
    DoomTrainer._attempt_load_state_dict(self.reward_decoder, saved_dict["reward_decoder"], "Reward Decoder")
    DoomTrainer._attempt_load_state_dict(self.action_model, saved_dict["action_model"], "Action Model")
    DoomTrainer._attempt_load_state_dict(self.discount_model, saved_dict["discount_model"], "Discount Model")
    DoomTrainer._attempt_load_state_dict(self.pos_ori_model, saved_dict["pos_ori_model"], "Position and Heading Model")
    if DoomTrainer._attempt_load_state_dict(self.value_model, saved_dict["value_model"], "Value Model"):
      self.target_value_model.load_state_dict(self.value_model.state_dict())
   

  def train_batch(self, train_metrics: Dict) -> Dict:
    
    model_losses = []
    kl_losses = []
    observation_kl_losses = []
    reward_losses = []
    observation_losses = []
    pos_ori_losses = []
    value_losses = []
    actor_losses = []
    prior_entropys = []
    posterior_entropys = []
    discount_losses = []
    mean_targets = []
    min_targets = []
    max_targets = []
    std_targets = []
    
    for _ in range(self.config.collect_intervals):
      observations, actions, positions, headings, rewards, terminals = self.replay_memory.sample()
      observations = torch.tensor(observations).to(self.device)              # t   to t+seq_len 
      actions      = torch.tensor(actions).to(self.device)                   # t-1 to t+seq_len-1
      positions    = torch.tensor(positions).to(self.device)                 # t   to t+seq_len 
      headings     = torch.tensor(headings).to(self.device).unsqueeze(-1)    # t   to t+seq_len 
      rewards      = torch.tensor(rewards).to(self.device).unsqueeze(-1)     # t-1 to t+seq_len-1
      nonterminals = torch.tensor(1-terminals).to(self.device).unsqueeze(-1) # t-1 to t+seq_len-1
      
      with torch.autocast(self.device.type):
        model_loss, kl_loss, observation_kl_loss, observation_loss, pos_ori_loss, reward_loss, discount_loss, prior_dist, posterior_dist, posterior = self.representation_loss(observations, actions, positions, headings, rewards, nonterminals)
      
      self.world_model_optimizer.zero_grad()
      model_loss.backward()
      self.world_model_optimizer.step()
      
      with torch.autocast(self.device.type):
        actor_loss, value_loss, target_info = self._actor_critic_loss(posterior)
      
      self.actor_model_optimizer.zero_grad()
      self.value_model_optimizer.zero_grad()
      actor_loss.backward()
      value_loss.backward()
      self.actor_model_optimizer.step()
      self.value_model_optimizer.step()
      
      with torch.no_grad():
        prior_entropy = torch.mean(prior_dist.entropy())
        posterior_entropy = torch.mean(posterior_dist.entropy())
        
      model_losses.append(model_loss.item())
      kl_losses.append(kl_loss.item())
      observation_kl_losses.append(observation_kl_loss.item())
      reward_losses.append(reward_loss.item())
      observation_losses.append(observation_loss.item())
      pos_ori_losses.append(pos_ori_loss.item())
      value_losses.append(value_loss.item())
      actor_losses.append(actor_loss.item())
      prior_entropys.append(prior_entropy.item())
      posterior_entropys.append(posterior_entropy.item())
      discount_losses.append(discount_loss.item())
      mean_targets.append(target_info['mean_target'])
      min_targets.append(target_info['min_target'])
      max_targets.append(target_info['max_target'])
      std_targets.append(target_info['std_target'])
      
      
    # Return a clusterfuck of training metrics
    train_metrics['model_loss']        = np.mean(model_losses)
    train_metrics['kl_loss']           = np.mean(kl_losses)
    train_metrics['obs_kl_loss']       = np.mean(observation_kl_losses)
    train_metrics['reward_loss']       = np.mean(reward_losses)
    train_metrics['observation_loss']  = np.mean(observation_losses)
    train_metrics['pos_ori_loss']      = np.mean(pos_ori_losses)
    train_metrics['value_loss']        = np.mean(value_losses)
    train_metrics['actor_loss']        = np.mean(actor_losses)
    train_metrics['prior_entropy']     = np.mean(prior_entropys)
    train_metrics['posterior_entropy'] = np.mean(posterior_entropys)
    train_metrics['discount_loss']     = np.mean(discount_losses)
    train_metrics['mean_target']       = np.mean(mean_targets)
    train_metrics['min_target']        = np.mean(min_targets)
    train_metrics['max_target']        = np.mean(max_targets)
    train_metrics['std_target']        = np.mean(std_targets)
    train_metrics['epsilon']           = self.action_model.epsilon

    return train_metrics
    
  def representation_loss(
    self, observations:torch.Tensor, actions:torch.Tensor, positions:torch.Tensor, 
    headings:torch.Tensor, rewards:torch.Tensor, nonterminals:torch.Tensor
  ):
    
    encoded_obs_dist = DiagonalGaussianDistribution(
      self.observation_encoder(observations), chunk_dim=2
    )
    embedded_observation = encoded_obs_dist.embedding(flatten_dim=2)

    prev_state = self.rssm.init_state(self.config.batch_size)
    prior, posterior = self.rssm.rollout_observation(
      self.config.seq_len, embedded_observation, actions, positions, headings, nonterminals, prev_state
    )
    post_modelstate = self.rssm.get_model_state(posterior)
    
    reconstructions = self.observation_decoder(post_modelstate[:-1])
    observation_loss = self.reconst_loss.reconstruction_nll_loss(observations[:-1], reconstructions)
    
    observation_kl_loss = encoded_obs_dist.kl()
    observation_kl_loss = torch.sum(observation_kl_loss) / np.prod(observation_kl_loss.shape[:-3])
    
    #pos_ori_dist = self.pos_ori_model.get_distribution(post_modelstate[:-1])
    pos_ori_cat = torch.cat((positions[:-1], headings[:-1]), dim=-1)
    pos_ori_loss = torch.nn.functional.mse_loss(self.pos_ori_model(post_modelstate[:-1]), pos_ori_cat, reduction='sum')
    #pos_ori_loss = -torch.mean(pos_ori_dist.log_prob(pos_ori_cat))
    
    reward_dist = self.reward_decoder.get_distribution(post_modelstate[:-1])
    reward_loss = self._reward_loss(reward_dist, rewards[1:])
    
    discount_dist = self.discount_model.get_distribution(post_modelstate[:-1])
    discount_loss = self._discount_loss(discount_dist, nonterminals[1:])
    
    prior_dist, posterior_dist, kl_loss = self._kl_loss(prior, posterior)
    
    model_loss = self.config.kl_loss_multiplier * kl_loss + self.config.obs_kl_loss_multiplier * observation_kl_loss + pos_ori_loss + reward_loss + observation_loss + self.config.discount_loss_multiplier * discount_loss
    return (
      model_loss, kl_loss, observation_kl_loss, observation_loss, pos_ori_loss,
      reward_loss, discount_loss, prior_dist, 
      posterior_dist, posterior#, reconstructions
    )
     
  def _kl_loss(self, prior: RSSMState, posterior: RSSMState) -> Tuple[td.Independent, td.Independent, torch.Tensor]:
    prior_dist     = self.rssm.get_distribution(prior)
    posterior_dist = self.rssm.get_distribution(posterior)
    alpha = self.config.kl_alpha
    kl_lhs = torch.mean(torch.distributions.kl.kl_divergence(
      self.rssm.get_distribution(self.rssm.detach_state(posterior)), prior_dist
    ))
    kl_rhs = torch.mean(torch.distributions.kl.kl_divergence(
      posterior_dist, self.rssm.get_distribution(self.rssm.detach_state(prior))
    ))
    kl_loss = alpha*kl_lhs + (1-alpha)*kl_rhs
    return prior_dist, posterior_dist, kl_loss
    
  def _reward_loss(self, reward_dist: td.Independent, rewards: torch.Tensor) -> torch.Tensor:
    return -torch.mean(reward_dist.log_prob(rewards))
    
  def _discount_loss(self, discount_dist: td.Independent, nonterminals: torch.Tensor) -> torch.Tensor:
    return -torch.mean(discount_dist.log_prob(nonterminals.float()))
  
  def _actor_critic_loss(self, posterior:RSSMState):
    with torch.autocast(self.device.type):
      with torch.no_grad():
        batched_posterior = self.rssm.detach_state(self.rssm.state_seq_to_batch(posterior, self.config.seq_len-1))
      with FreezeParameters(self.world_models):
        imagined_states, imagined_log_prob, policy_entropy = self.rssm.rollout_imagination(
          self.config.horizon, self.action_model, self.pos_ori_model, batched_posterior
        )
    
      imagined_modelstates = self.rssm.get_model_state(imagined_states)
      with FreezeParameters(self.world_models + self.value_models + [self.target_value_model] + [self.discount_model]):
        imagined_reward_dist = self.reward_decoder.get_distribution(imagined_modelstates)
        imagined_reward = imagined_reward_dist.mean
        imagined_value_dist = self.target_value_model.get_distribution(imagined_modelstates)
        imagined_value = imagined_value_dist.mean
        discount_dist  = self.discount_model.get_distribution(imagined_modelstates)
        discount_values = self.config.discount * torch.round(discount_dist.base_dist.probs)

      actor_loss, discount, lambda_returns = self._actor_loss(
        imagined_reward, imagined_value, discount_values, imagined_log_prob, policy_entropy
      )
      value_loss = self._value_loss(imagined_modelstates, discount, lambda_returns)
      
      mean_target = torch.mean(lambda_returns, dim=1)
      target_info = {
        'min_target' : torch.min(mean_target).item(),
        'max_target' : torch.max(mean_target).item(),
        'std_target' : torch.std(mean_target).item(),
        'mean_target': torch.mean(mean_target).item(),
      }
    
    return actor_loss, value_loss, target_info
  
  def _actor_loss(self, imagined_reward, imagined_value, discount_values, imagined_log_prob, policy_entropy):  
    lambda_returns = compute_return(
      imagined_reward[:-1], imagined_value[:-1], discount_values[:-1], 
      bootstrap=imagined_value[-1], td_lambda=self.config.td_lambda
    ).to(self.device)
    advantage = (lambda_returns-imagined_value[:-1]).detach()
    objective = imagined_log_prob[1:].unsqueeze(-1) * advantage
    
    discount_values = torch.cat([torch.ones_like(discount_values[:1]), discount_values[1:]])
    discount = torch.cumprod(discount_values[:-1], dim=0, dtype=torch.float16)
    policy_entropy = policy_entropy[1:].unsqueeze(-1)
    actor_loss = -torch.sum(torch.mean(discount * (objective + self.config.actor_entropy_scale * policy_entropy), dim=1))
    return actor_loss, discount, lambda_returns
    
  def _value_loss(self, imagined_modelstates, discount, lambda_returns):
    with torch.no_grad():
      value_modelstates = imagined_modelstates[:-1].detach()
      value_discount = discount.detach()
      value_target = lambda_returns.detach()
      
    value_dist = self.value_model.get_distribution(value_modelstates)
    return -torch.mean(value_discount * value_dist.log_prob(value_target).unsqueeze(-1))
  
def compute_return(
  reward: torch.Tensor, value: torch.Tensor, discount: torch.Tensor, bootstrap: torch.Tensor, td_lambda: float
):
  """
  Compute the discounted reward for a batch of data.
  reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
  Bootstrap is [batch, 1]
  """
  next_values = torch.cat([value[1:], bootstrap[None]], 0)
  target = reward + discount * next_values * (1 - td_lambda)
  outputs = []
  accumulated_reward = bootstrap
  # Iterate through timesteps backwards, from reward.shape[0]-1 to 0
  for t in range(reward.shape[0] - 1, -1, -1):
    accumulated_reward = target[t] + discount[t] * td_lambda * accumulated_reward
    outputs.append(accumulated_reward)
  returns = torch.flip(torch.stack(outputs), [0])
  return returns


