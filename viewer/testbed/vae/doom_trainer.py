
from typing import Dict, Tuple

import torch
import torch.optim as optim
import torch.distributions as td

from replay_memory import ReplayMemory
from doom_rssm import DoomRSSM, RSSMState
from doom_models import DoomDenseModel, DoomDiscreteActionModel
from conv_encode_decode import ConvEncoder, ConvDecoder

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
      
    embedding_size   = config.embedding_size
    modelstate_size  = stoch_size + deter_size
    
    self.replay_memory = ReplayMemory(config.capacity, observation_shape, action_size, config.seq_len, config.batch_size)
    
    self.rssm = DoomRSSM(action_size, embedding_size, self.device, config.rssm_info).to(self.device)
    self.action_model = DoomDiscreteActionModel(action_size, deter_size, stoch_size, config.actor_info, config.epsilon_info).to(self.device)
    self.reward_decoder = DoomDenseModel((1,), modelstate_size, config.reward_info).to(self.device)
    self.value_model = DoomDenseModel((1,), modelstate_size, config.critic_info).to(self.device)
    self.target_value_model = DoomDenseModel((1,), modelstate_size, config.critic_info).to(self.device)
    self.target_value_model.load_state_dict(self.value_model.state_dict()) # Load the same initial weights as the value model
    
    self.discount_model = DoomDenseModel((1,), modelstate_size, config.discount_info).to(self.device)
    
    self.observation_encoder = ConvEncoder(observation_shape, embedding_size).to(self.device)
    self.observation_decoder = ConvDecoder(
      observation_shape, self.observation_encoder.conv_output_shape, 
      self.observation_encoder.output_sizes, modelstate_size
    ).to(self.device)
     
  def _optimizer_init(self, config):
    model_lr = config.lr_info['model']
    actor_lr = config.lr_info['actor']
    value_lr = config.lr_info['critic']
    
    self.world_models = [
      self.observation_encoder, self.observation_decoder, 
      self.rssm, self.reward_decoder, self.discount_model
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
      action = env.action_space.sample()
      next_observation, reward, done = env.step(action)
      if done:
        self.replay_memory.add(observation, action, reward, done)
        observation, done = env.reset(), False
      else:
        self.replay_memory.add(observation, action, reward, done)
        observation = next_observation
  
  def train_batch(self) -> Dict:
    observations, actions, rewards, terminals = self.replay_memory.sample()
    observations = torch.tensor(observations, dtype=torch.float32).to(self.device)              # t   to t+seq_len 
    actions      = torch.tensor(actions, dtype=torch.float32).to(self.device)                   # t-1 to t+seq_len-1
    rewards      = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(-1)     # t-1 to t+seq_len-1
    nonterminals = torch.tensor(1-terminals, dtype=torch.float32).to(self.device).unsqueeze(-1) # t-1 to t+seq_len-1
    
    # TODO: SPLIT THIS UP!
    model_loss, kl_loss, observation_loss, reward_loss, discount_loss, prior_dist, posterior_dist, posterior, reconstructions = self.representation_loss(observations, actions, rewards, nonterminals)
    
    self.world_model_optimizer.zero_grad()
    model_loss.backward()
    self.world_model_optimizer.step()
    
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
    
    # Return a clusterfuck of training metrics
    return {
      'model_loss': model_loss,
      'kl_loss': kl_loss,
      'reward_loss': reward_loss,
      'observation_loss': observation_loss,
      'value_loss': value_loss,
      'actor_loss': actor_loss,
      'prior_entropy': prior_entropy,
      'posterior_entropy': posterior_entropy,
      'discount_loss': discount_loss,
      'target_info': target_info
    }
    
  def representation_loss(self, observations:torch.Tensor, actions:torch.Tensor, rewards:torch.Tensor, nonterminals:torch.Tensor):
    embedded_observation = self.observation_encoder(observations)
    prev_state = self.rssm.init_state(self.config.batch_size)
    prior, posterior = self.rssm.rollout_observation(
      self.config.seq_len, embedded_observation, actions, nonterminals, prev_state
    )
    post_modelstate = self.rssm.get_model_state(posterior)
    
    reconstructions = self.observation_decoder(post_modelstate[:-1])
    observation_dist = self.observation_decoder.get_distribution(observations[:-1], reconstructions)
    observation_loss = self._observation_loss(observation_dist, observations[:-1])
    
    reward_dist = self.reward_decoder.get_distribution(post_modelstate[:-1])
    reward_loss = self._reward_loss(reward_dist, rewards[1:])
    
    discount_dist = self.discount_model.get_distribution(post_modelstate[:-1])
    discount_loss = self._discount_loss(discount_dist, nonterminals[1:])
    
    prior_dist, posterior_dist, kl_loss = self._kl_loss(prior, posterior)
    
    model_loss = self.config.kl_loss_multiplier * kl_loss + reward_loss + observation_loss + self.config.discount_loss_multiplier * discount_loss
    return (
      model_loss, kl_loss, observation_loss, 
      reward_loss, discount_loss, prior_dist, 
      posterior_dist, posterior, reconstructions
    )
    
  def _observation_loss(self, observation_dist: td.Independent, observations: torch.Tensor) -> torch.Tensor:
    return -torch.mean(observation_dist.log_prob(observations))
  
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
    with torch.no_grad():
      batched_posterior = self.rssm.detach_state(self.rssm.state_seq_to_batch(posterior, self.config.seq_len-1))
    with FreezeParameters(self.world_models):
      imagined_states, imagined_log_prob, policy_entropy = self.rssm.rollout_imagination(
        self.config.horizon, self.action_model, batched_posterior
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
    )
    advantage = (lambda_returns-imagined_value[:-1]).detach()
    objective = imagined_log_prob[1:].unsqueeze(-1) * advantage
    
    discount_values = torch.cat([torch.ones_like(discount_values[:1]), discount_values[1:]])
    discount = torch.cumprod(discount_values[:-1], 0)
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