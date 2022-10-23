import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as T
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import vizdoom as vzd
import vizdoomgym
VIZDOOM_GAME_PATH = "_vizdoom"

from stable_baselines3.common.atari_wrappers import (  # isort:skip
  ClipRewardEnv,
  MaxAndSkipEnv,
)

from net_init import fc_layer_init, conv_layer_init
from sd_conv import SDEncoder

from gym import ObservationWrapper
class DoomObservation(ObservationWrapper):
  def __init__(self, env, shape) -> None:
    super().__init__(env)
    self.shape = shape
    obs_shape = (self.observation_space.shape[-1],) + self.shape
    from gym.spaces import Box
    self.observation_space = Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)
  
  def permute_observation(self, observation):
    observation = np.transpose(observation, (2,0,1))
    observation = torch.tensor(observation.copy(), dtype=torch.float)
    return observation
  
  def observation(self, observation):
    observation = self.permute_observation(observation)
    return T.Resize(self.shape)(observation).squeeze(0).numpy() / 255.0

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="", help="Preexisting model to load (.chkpt file)")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="VizdoomCorridor-v0", help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4, help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--save-timesteps", type=int, default=50000, help="Timesteps between network saves")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Game specific arguments
    parser.add_argument("--multidiscrete-actions", type=bool, default=True, 
      help="Whether the agent uses multidiscrete actions (up to 2 at a time) or not") # NOTE: Multidiscrete converges faster!
    parser.add_argument("--smart-actions", type=bool, default=True,
      help="Whether to use smart multdiscrete actions (e.g., you don't move left and right simulataneously), this limits actions to 2 at a time.")


    # Network specific arguments
    parser.add_argument("--net-output-size", type=int, default=2048, help="Output size of the convolutional network, input size to the LSTM")
    parser.add_argument("--lstm-hidden-size", type=int, default=512, help="Hidden size of the LSTM")
    parser.add_argument("--sd-conv", type=bool, default=True, help="Use the Stable Diffusion encoder residual convolutional network")
    parser.add_argument("--obs-shape", type=str, default="60,80", # 69,92 doesn't appear to help convergence... 
      help="Shape of the RGB screenbuffer (height, width) after being processed (when fed to the convnet).")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8, help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1, help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None, help="the target KL divergence threshold")
    
    args = parser.parse_args()
    if args.smart_actions:
      args.multidiscrete_actions = False
      
    args.obs_shape = tuple([int(item) for item in args.obs_shape.split(',')])
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args

def make_env(args, seed, idx, run_name):
  def thunk():
    max_buttons_pressed = 0 if args.multidiscrete_actions else 1

    env = gym.make(
      args.gym_id, 
      set_window_visible=(idx==0), 
      game_path=os.path.join(VIZDOOM_GAME_PATH, "doom.wad"),
      resolution=vzd.ScreenResolution.RES_640X480,
      max_buttons_pressed=max_buttons_pressed,
      smart_actions=args.smart_actions
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if args.capture_video and idx == 0:
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    env = MaxAndSkipEnv(env, skip=4)
    env = ClipRewardEnv(env)
    env = DoomObservation(env, shape=args.obs_shape)
    #env = gym.wrappers.ResizeObservation(env, (84, 84))
    #env = gym.wrappers.GrayScaleObservation(env)
    #env = gym.wrappers.FrameStack(env, 1)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

  return thunk

class Agent(nn.Module):
  def __init__(self, args, envs):
    super(Agent, self).__init__()
    
    net_output_size  = args.net_output_size  # Original was 512
    lstm_hidden_size = args.lstm_hidden_size # Original was 128
    
    if args.sd_conv:
      self.network = SDEncoder(args, envs)
    else:
      out_channel_list = [32, 64, 64] # Original was [32, 64, 64]
      kernel_size_list = [ 8,  4,  3] # Original was [ 8,  4,  3]
      stride_list      = [ 4,  2,  1] # Original was [ 4,  2,  1]
      padding_list     = [ 0,  0,  0] # Original was [ 0,  0,  0]
      
      self.network = nn.Sequential()
      curr_channels, curr_height, curr_width = envs.single_observation_space.shape
      #w_to_h_ratio = curr_width / curr_height
      
      for out_channels, kernel_size, stride, padding in zip(out_channel_list, kernel_size_list, stride_list, padding_list):
        # NOTE: Square vs. Rectangular kernels appear to have no noticable effect
        # If anything, rectangular is worse. Use square kernels for simplicity.
        kernel_size_h = kernel_size
        kernel_size_w = kernel_size #round(w_to_h_ratio*kernel_size)
        self.network.append(conv_layer_init(
          nn.Conv2d(curr_channels, out_channels, (kernel_size_h, kernel_size_w), stride)
        ))
        self.network.append(nn.ELU(inplace=True))
        
        curr_width  = int((curr_width-kernel_size_w + 2*padding) / stride + 1)
        curr_height = int((curr_height-kernel_size_h + 2*padding) / stride + 1)
        curr_channels = out_channels
        
      self.network.append(nn.Flatten())
      conv_output_size = curr_width*curr_height*curr_channels
      self.network.append(fc_layer_init(nn.Linear(conv_output_size, net_output_size)))
      self.network.append(nn.LeakyReLU(0.25, inplace=True)) # LeakyReLU with a higher slope here works best vs. ReLU/SELU/ELU

    # NOTE: LSTM appears to like a 4:1 ratio of input size to hidden size, increasing
    # this ratio is detrimental to the network (doesn't converge or takes a very long time)
    self.lstm = nn.LSTM(net_output_size, lstm_hidden_size)
    for name, param in self.lstm.named_parameters():
      if "bias" in name:
        nn.init.constant_(param, 0)
      elif "weight" in name:
        nn.init.orthogonal_(param, 1.0)
        
    self.multidiscrete_actions = args.multidiscrete_actions
    if self.multidiscrete_actions:
      self.nvec = envs.single_action_space.nvec
      action_space_size = self.nvec.sum()
    else:
      action_space_size = envs.single_action_space.n
            
    self.actor = fc_layer_init(nn.Linear(lstm_hidden_size, action_space_size), std=0.01)
    self.critic = fc_layer_init(nn.Linear(lstm_hidden_size, 1), std=1)

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
        (
          (1.0 - d).view(1, -1, 1) * lstm_state[0],
          (1.0 - d).view(1, -1, 1) * lstm_state[1],
        ),
      )
      new_hidden += [h]
    new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
    return new_hidden, lstm_state

  def get_value(self, x, lstm_state, done):
    hidden, _ = self.get_states(x, lstm_state, done)
    return self.critic(hidden)

  def get_action_and_value(self, x, lstm_state, done, action=None):
    hidden, lstm_state = self.get_states(x, lstm_state, done)
    logits = self.actor(hidden)
    
    if self.multidiscrete_actions:
      split_logits = torch.split(logits, self.nvec.tolist(), dim=1)
      multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
      if action is None:
        action = torch.stack([categorical.sample() for categorical in multi_categoricals])
      logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
      entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
      return action.T, logprob.sum(0), entropy.sum(0), self.critic(hidden), lstm_state
    else:
      probs = Categorical(logits=logits)
      if action is None:
          action = probs.sample()
      return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


if __name__ == "__main__":
  args = parse_args()
  
  run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
  run_dir = os.path.join("runs", run_name)
  
  if args.track:
      import wandb
      wandb.init(
          project=args.wandb_project_name,
          entity=args.wandb_entity,
          sync_tensorboard=True,
          config=vars(args),
          name=run_name,
          monitor_gym=True,
          save_code=True,
      )
  writer = SummaryWriter(run_dir)
  writer.add_text(
      "hyperparameters",
      "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
  )

  # TRY NOT TO MODIFY: seeding
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = args.torch_deterministic

  device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

  # env setup
  envs = gym.vector.SyncVectorEnv(
      [make_env(args, args.seed + i, i, run_name) for i in range(args.num_envs)]
  )
  
  assert args.multidiscrete_actions and isinstance(envs.single_action_space, gym.spaces.MultiDiscrete) or \
    not args.multidiscrete_actions and isinstance(envs.single_action_space, gym.spaces.Discrete), \
    "Discrete/MultiDiscrete action space mismatch!"

  agent = Agent(args, envs).to(device)
  optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

  # ALGO Logic: Storage setup
  obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
  actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
  logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
  rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
  dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
  values = torch.zeros((args.num_steps, args.num_envs)).to(device)

  # TRY NOT TO MODIFY: start the game
  global_step = 0
  
  if args.model is not None and len(args.model) > 0:
    if os.path.exists(args.model):
      print(f"Model file '{args.model}' found, loading...")
      model_dict = torch.load(args.model)
      init_step = model_dict["timesteps"]
      global_step = init_step
      args.total_timesteps += global_step
      try:
        agent.load_state_dict(model_dict["agent"], strict=False)
      except RuntimeError as e:
        print("Could not load agent networks:")
        print(e)
      optimizer.load_state_dict(model_dict["optim"])
      print("Model loaded!")
    else:
      print(f"Could not find/load model file '{args.model}'")
    
  
  start_time = time.time()
  next_obs = torch.Tensor(envs.reset()).to(device)
  next_done = torch.zeros(args.num_envs).to(device)
  next_lstm_state = (
      torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
      torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
  )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
  num_updates = args.total_timesteps // args.batch_size
  cum_scores = np.zeros(args.num_envs, dtype=np.float64)

  for update in range(1, num_updates + 1):
      initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
      # Annealing the rate if instructed to do so.
      if args.anneal_lr:
          frac = 1.0 - (update - 1.0) / num_updates
          lrnow = frac * args.learning_rate
          optimizer.param_groups[0]["lr"] = lrnow

      for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        
        if (global_step // args.num_envs) % args.save_timesteps == 0:
          save_dict = {
            "timesteps": global_step,
            "agent": agent.state_dict(),
            "optim": optimizer.state_dict()
          }
          save_path = os.path.join(run_dir, f"doom_ppo_agent_{global_step}.chkpt")
          torch.save(save_dict, save_path)     
        
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
          action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
          values[step] = value.flatten()
        actions[step] = action
        logprobs[step] = logprob

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        if global_step % 1000 == 0:
          writer.add_image("images/observation", next_obs[0], global_step)

        for env_idx, item in enumerate(info):
          if "episode" in item.keys():
            cum_scores[env_idx] += item["episode"]["r"]
            print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
            writer.add_scalar("charts/episodic_return",   item["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length",   item["episode"]["l"], global_step)
            writer.add_scalar("charts/cumulative_return", cum_scores[env_idx],  global_step)
            break

      # bootstrap value if not done
      with torch.no_grad():
        next_value = agent.get_value(
            next_obs,
            next_lstm_state,
            next_done,
        ).reshape(1, -1)
        if args.gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
          returns = torch.zeros_like(rewards).to(device)
          for t in reversed(range(args.num_steps)):
              if t == args.num_steps - 1:
                  nextnonterminal = 1.0 - next_done
                  next_return = next_value
              else:
                  nextnonterminal = 1.0 - dones[t + 1]
                  next_return = returns[t + 1]
              returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
          advantages = returns - values

      # flatten the batch
      b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
      b_logprobs = logprobs.reshape(-1)
      b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
      b_dones = dones.reshape(-1)
      b_advantages = advantages.reshape(-1)
      b_returns = returns.reshape(-1)
      b_values = values.reshape(-1)

      # Optimizing the policy and value network
      assert args.num_envs % args.num_minibatches == 0
      envsperbatch = args.num_envs // args.num_minibatches
      envinds = np.arange(args.num_envs)
      flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
      clipfracs = []
      for epoch in range(args.update_epochs):
          np.random.shuffle(envinds)
          for start in range(0, args.num_envs, envsperbatch):
              end = start + envsperbatch
              mbenvinds = envinds[start:end]
              mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

              _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                  b_obs[mb_inds],
                  (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                  b_dones[mb_inds],
                  b_actions.long()[mb_inds].T if args.multidiscrete_actions else b_actions.long()[mb_inds],
              )
              logratio = newlogprob - b_logprobs[mb_inds]
              ratio = logratio.exp()

              with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

              mb_advantages = b_advantages[mb_inds]
              if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

              # Policy loss
              pg_loss1 = -mb_advantages * ratio
              pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
              pg_loss = torch.max(pg_loss1, pg_loss2).mean()

              # Value loss
              newvalue = newvalue.view(-1)
              if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
              else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

              entropy_loss = entropy.mean()
              loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

              optimizer.zero_grad()
              loss.backward()
              nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
              optimizer.step()

          if args.target_kl is not None:
            if approx_kl > args.target_kl:
              break

      y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
      var_y = np.var(y_true)
      explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

      # TRY NOT TO MODIFY: record rewards for plotting purposes
      writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
      writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
      writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
      writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
      writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
      writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
      writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
      writer.add_scalar("losses/explained_variance", explained_var, global_step)
      print("SPS:", int(global_step / (time.time() - start_time)))
      writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

  envs.close()
  writer.close()