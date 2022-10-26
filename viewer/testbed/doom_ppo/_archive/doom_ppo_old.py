import os
import argparse
import time
import random
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from doom_agent import DoomAgent
from doom_env import DoomEnv, DEFAULT_PREPROCESS_FINAL_SHAPE
from env_vec import EnvVec

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, default=None, help="Preexisting model to load (.chkpt file)")
  parser.add_argument("--learning-rate", type=float, default=1e-4,
    help="Learning rate of the optimizer")
  parser.add_argument("--save-timesteps", type=int, default=50000, help="Timesteps between network saves")
  parser.add_argument("--total-timesteps", type=int, default=100000000, help="Total timesteps of all training")
  parser.add_argument("--seed", type=int, default=1, help="RNG seed")
  parser.add_argument("--torch-deterministic", type=lambda x:bool(strtobool(x)), 
    default=True, nargs="?", const=True, help="If toggled, `torch.backends.cudnn.deterministic=False`")
  parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?",
    const=True, help="If toggled, this experiment will be tracked with Weights and Biases")
  
  # Network specific args
  parser.add_argument("--net-output-size", type=int, default=4096, help="Output size of the convolutional network, input size to the LSTM")
  parser.add_argument("--lstm-hidden-size", type=int, default=1024, help="Hidden size of the LSTM")
  parser.add_argument("--use-classifier", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
    help="Whether or not to use the classifier network")
  parser.add_argument("--classifier-num-hidden", type=int, default=1, help="Number of hidden layers for the classifier network")
  parser.add_argument("--classifier-hidden-size", type=int, default=512, help="Number of nodes in each hidden layer for the classifier network")
  parser.add_argument("--actor-critic-num-hidden", type=int, default=0, help="Number of hidden layers for the actor and critic networks")
  parser.add_argument("--actor-critic-hidden-size", type=int, default=1024, help="Number of nodes in each hidden layer for the actor and critic networks")
  parser.add_argument("--conv-channels", type=str, default="32,64,64", help="Set of output channels of the conv2d network")
  parser.add_argument("--conv-kernels", type=str, default="8,4,3", help="Set of kernel sizes of the conv2d network")
  parser.add_argument("--conv-strides", type=str, default="4,2,1", help="Set of stride sizes of the conv2d network")
  parser.add_argument("--conv-paddings", type=str, default="0,0,0", help="Set of padding sizes of the conv2d network")
  parser.add_argument("--use-labels-buffer", type=lambda x: bool(strtobool(x)), default=True,
    help="Toggle use of the label buffer for in-game object identification to help the agent learn faster")
  
  # Game specific args
  parser.add_argument("--scenario-name", type=str, default="basic", help="Scenario (don't include '.wad') to load to train on.")
  parser.add_argument("--map", type=str, default="map01", help="Map(s) to load for play/training")
  parser.add_argument("--clip-reward", type=lambda x: bool(strtobool(x)), default=True, help="Clip each step's reward to -1, 0, or 1 (i.e., sign(reward)")
  
  # Algorithm specific args
  parser.add_argument("--num-envs", type=int, default=32, help="The number of parallel game environments") # Implement this and change to 4?
  parser.add_argument("--num-steps", type=int, default=256, help="The number of steps per policy rollout")
  parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    help="Toggle learning rate annealing for the policy and value networks")
  parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
    help="Use GAE for advantage computation")
  parser.add_argument("--gamma", type=float, default=0.99, help="The discount factor gamma")
  parser.add_argument("--gae-lambda", type=float, default=0.95,
      help="The lambda for the general advantage estimation")
  parser.add_argument("--num-minibatches", type=int, default=4, help="The number of mini-batches")
  parser.add_argument("--update-epochs", type=int, default=8, help="The K epochs to update the policy")
  parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
  parser.add_argument("--clip-coef", type=float, default=0.2, help="The surrogate clipping coefficient")
  parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
      help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
  parser.add_argument("--ent-coef", type=float, default=0.01, help="Coefficient of the entropy loss")
  parser.add_argument("--vf-coef", type=float, default=0.5, help="Coefficient of the value loss function")
  parser.add_argument("--cf-coef", type=float, default=0.5, help="Coefficient of the classification loss function")
  parser.add_argument("--max-grad-norm", type=float, default=0.5, help="The maximum norm for the gradient clipping")
  parser.add_argument("--target-kl", type=float, default=None, help="The target KL divergence threshold")
  
  args = parser.parse_args()
  args.batch_size = int(args.num_envs * args.num_steps)
  args.minibatch_size = int(args.batch_size // args.num_minibatches)
  args.conv_channels = [int(item) for item in args.conv_channels.split(',')]
  args.conv_kernels  = [int(item) for item in args.conv_kernels.split(',')]
  args.conv_strides  = [int(item) for item in args.conv_strides.split(',')]
  args.conv_paddings = [int(item) for item in args.conv_paddings.split(',')]
  args.observation_shape = DEFAULT_PREPROCESS_FINAL_SHAPE
  
  return args

if __name__ == "__main__":
  args = parse_args()
  run_name = f"doom_ppo_{args.seed}_{args.scenario_name}_{int(time.time())}"
  if args.track:
    import wandb
    wandb.init(
      project="doom_ppo",
      config=vars(args),
      name=run_name,
      save_code=True,
      sync_tensorboard=True
    )
    
  run_dir = os.path.join("runs", run_name)
  writer = SummaryWriter(run_dir)
  writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
  )
  
  # Seeding
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = args.torch_deterministic
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Setup the vizdoom environment wrapper, agent, optimizer
  envs = EnvVec([DoomEnv(args, 1000000, i == 0) for i in range(args.num_envs)])
  agent = DoomAgent(envs, args).to(device)
  optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
  
  label_loss_crit = nn.BCEWithLogitsLoss()
  
  # Storage setup
  obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_shape()).to(device)
  labels = torch.zeros((args.num_steps, args.num_envs) + envs.label_shape()).to(device)
  actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
  logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
  rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
  dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
  values = torch.zeros((args.num_steps, args.num_envs)).to(device)

  # Start game
  init_step = 0
  global_step = 0
  if args.model is not None:
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
  
  cum_scores = np.zeros(args.num_envs, dtype=np.float64)
  start_time = time.time()
  raw_obs, raw_labels = envs.reset()
  next_obs    = torch.Tensor(raw_obs).to(device)
  #next_labels = torch.Tensor(raw_labels).to(device)
  next_done   = torch.zeros(args.num_envs).to(device)
  
  next_lstm_state = (
    torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device)
  ) # Hidden and cell states
  num_updates = args.total_timesteps // args.batch_size
  
  for update in range(1, num_updates+1):
    initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
    if args.anneal_lr:
      frac = 1.0 - (update - 1.0) / num_updates
      new_lr = frac * args.learning_rate
      optimizer.param_groups[0]['lr'] = new_lr

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
      
      obs[step]    = next_obs
      #labels[step] = next_labels
      dones[step]  = next_done
      
      # Action logic
      with torch.no_grad():
        action, logprob, _, value, next_lstm_state = agent.get_action_and_value(next_obs, next_lstm_state, next_done)
        values[step] = value.flatten()
      actions[step]  = action
      logprobs[step] = logprob
      
      # Execute the game and log data
      next_obs, next_labels, reward, done, info = envs.step(action.cpu().numpy())
      rewards[step] = torch.tensor(reward).to(device).view(-1)
      next_obs    = torch.Tensor(next_obs).to(device)
      #next_labels = torch.Tensor(next_labels).to(device)
      next_done   = torch.Tensor(done).to(device)
    
      for env_idx, item in enumerate(info):
        if "episode" in item.keys():
          cum_scores[env_idx] += item["episode"]["r"]
          print(f"env_idx={env_idx}, global_step={global_step}, episodic_return={item['episode']['r']}, cumulative_return={cum_scores[env_idx]}")
          writer.add_scalar("charts/episodic_return",   item["episode"]["r"], global_step)
          writer.add_scalar("charts/episodic_length",   item["episode"]["l"], global_step)
          writer.add_scalar("charts/cumulative_return", cum_scores[env_idx],  global_step)
          break
    
    # Bootstrap value if not done
    with torch.no_grad():
      next_value = agent.get_value(next_obs, next_lstm_state, next_done).reshape(1, -1)
      if args.gae:
        advantages = torch.zeros_like(rewards).to(device)
        last_gae_lambda = 0
        for t in reversed(range(args.num_steps)):
          if t == args.num_steps-1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
          else:
            nextnonterminal = 1.0 - dones[t+1]
            nextvalues = values[t+1]
          delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
          advantages[t] = last_gae_lambda = delta + args.gamma * args.gae_lambda * nextnonterminal * last_gae_lambda
        returns = advantages + values
        
      else:
        returns = torch.zeros_like(rewards).to(device)
        for t in reversed(range(args.num_steps)):
          if t == args.num_steps-1:
            nextnonterminal = 1.0 - next_done
            next_return = next_value
          else:
            nextnonterminal = 1.0 - dones[t+1]
            next_return = returns[t+1]
          returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
        advantages = returns - values
        
    # Flatten the batch 
    b_obs        = obs.reshape((-1,) + envs.observation_shape())
    #b_labels     = labels.reshape((-1,) + envs.label_shape())
    b_logprobs   = logprobs.reshape(-1)
    b_actions    = actions.reshape((-1,1))
    b_dones      = dones.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns    = returns.reshape(-1)
    b_values     = values.reshape(-1)
    
    # Optimize the policy and value network
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
        mb_inds = flatinds[:, mbenvinds].ravel() # Be really careful about the index
        
        _, newlogprob, entropy, newvalue, _, class_logits = agent.get_action_value_classify(
          b_obs[mb_inds],
          (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
          b_dones[mb_inds],
          b_actions.long()[mb_inds]
        )
        logratio = newlogprob - b_logprobs[mb_inds]
        ratio = logratio.exp()
        
        with torch.no_grad():
          # Calculate the approx. KL
          old_approx_kl = (-logratio).mean()
          approx_kl = ((ratio-1) - logratio).mean()
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
          v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef,)
          v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
          v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
          v_loss = 0.5 * v_loss_max.mean()
        else:
          v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
        
        # Classification loss
        #if class_logits is None:
        #c_loss = 0
        #else:
        #  c_loss = label_loss_crit(class_logits, b_labels[mb_inds])
        
        # Entropy loss
        entropy_loss = entropy.mean()
        
        # Total loss
        loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss #+ args.cf_coef * c_loss
        
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
    
    # Record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    #if args.use_classifier:
    #  writer.add_scalar("losses/classification_loss", c_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    #print("SPS:", int((global_step-init_step) / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    
  envs.close()
  writer.close()