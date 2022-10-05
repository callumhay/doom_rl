import argparse
import datetime
import os

import wandb
import torch
from torchsummary import summary
import numpy as np
import pandas as pd

from config import Config, CHECKPOINT_DIR
from doom_env import DoomEnv, PREPROCESS_FINAL_SHAPE_C_H_W
from diagonal_gaussian_distribution import DiagonalGaussianDistribution

from doom_trainer import DoomTrainer

def main(args):
  os.makedirs(CHECKPOINT_DIR, exist_ok=True) # Setup checkpoints directory
  
  start_time = datetime.datetime.now()
  start_time_str = start_time.strftime("%m-%d-%Y_%H:%M:%S")
  
  if torch.cuda.is_available() and args.device == 'cuda':
    device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
  else:
    device = torch.device('cpu')
    
  #torch.autograd.set_detect_anomaly(True)
  
  env = DoomEnv(args.map, args.episode_max_steps)

  train_steps = 0
  start_episode = 1
  csv_filepath = os.path.join(CHECKPOINT_DIR, f"training_log_{start_time_str}.csv")
  model_dict = None
  
  if args.model != None:
    if os.path.exists(args.model):
      print(f"Model file '{args.model}' found, loading...")
      model_dict = torch.load(args.model)
      print(f"Model file loaded!")
      train_steps = model_dict["train_steps"]
      if 'episode' in model_dict:
        start_episode = model_dict["episode"]
      #prev_csv_filepath = model_dict["csv_filepath"]
      #if os.path.exists(prev_csv_filepath):
      #  print(f"Found previous training metrics csv file '{prev_csv_filepath}', renaming to '{csv_filepath}' ...")
      #  os.rename(prev_csv_filepath, csv_filepath)
    else:
      print(f"Could not find/load model file '{args.model}'")

  config = Config(
    observation_shape=PREPROCESS_FINAL_SHAPE_C_H_W,
    action_size=env.action_size,
    seq_len=args.seq_len,
    batch_size=args.batch_size,
    train_episodes=args.num_episodes,
    train_every_steps=args.seq_len,
    explore_steps=args.explore_steps,
    save_every_steps=args.save_steps,
    csv_filepath=csv_filepath,
  )
  config.epsilon_info['start_epsilon'] = args.epsilon
  
  trainer = DoomTrainer(config, device)
  if model_dict != None:
    trainer.load_save_dict(model_dict)
 
  #summary(trainer.observation_encoder, input_size=PREPROCESS_FINAL_SHAPE_C_H_W, device=device.type)
  #modelstate_size = config.modelstate_size
  #summary(trainer.observation_decoder, input_size=(modelstate_size,), device=device.type)
  
  with wandb.init(project='Doom Dreamer RL', config=config.__dict__):
    
    trainer.initial_exploration(env)
    train_metrics = {}
    total_score = 0
    for episode in range(start_episode, config.train_episodes+start_episode):
      
      done = False
      observation, score = env.reset(), 0
      episode_actor_entopys = []
      prev_action = torch.zeros(1, env.action_size, device=device)
      prev_position = torch.zeros(1, 3, device=device)
      prev_heading = torch.zeros(1, 1, device=device)
      prev_state = trainer.rssm.init_state(1)
      
      while not done:
        train_steps += 1
        
        if train_steps % config.train_every_steps == 0:
          train_metrics = trainer.train_batch(train_metrics)
        if train_steps % config.slow_target_update_steps == 0:
          trainer.update_target()
        if train_steps % config.save_every_steps == 0:
          trainer.save_model(train_steps, episode)
        
        # Choose the next action based on the state of the current network and a epsilon greedy policy
        with torch.autocast(device.type):
          with torch.no_grad():
            encoded_obs_dist = DiagonalGaussianDistribution(
              trainer.observation_encoder(torch.tensor(observation, device=device).unsqueeze(0)),
              chunk_dim=1
            )
            observation_embed = encoded_obs_dist.embedding(flatten_dim=1)
            _, posterior_state = trainer.rssm.observe(observation_embed, prev_action, prev_position, prev_heading, not done, prev_state)
            modelstate = trainer.rssm.get_model_state(posterior_state)
            action, action_dist = trainer.action_model(modelstate)
            action = trainer.action_model.add_exploration(action).detach()
            action_entropy = torch.mean(action_dist.entropy()).item()
            episode_actor_entopys.append(action_entropy)
      
        next_observation, reward, done = env.step(action.squeeze(0).cpu().numpy())
        score += reward
        
        position = env.player_pos()
        heading  = env.player_heading()
        trainer.replay_memory.add(observation, action.squeeze(0).detach().cpu().numpy(), position, heading, reward, done)

        if not done:
          observation = next_observation
          prev_state  = posterior_state
          prev_action = action
          prev_position = torch.tensor(position, dtype=torch.float32, device=device).unsqueeze(0)
          prev_heading  = torch.tensor([heading], dtype=torch.float32, device=device).unsqueeze(0)
      
      total_score += score
      train_metrics['total_reward']  = total_score
      train_metrics['train_rewards'] = score
      train_metrics['action_ent']    = np.mean(episode_actor_entopys)
      train_metrics['train_steps']   = train_steps
      wandb.log(train_metrics, step=episode)
      
      #dataframe = pd.DataFrame(train_metrics, index=[episode])
      #dataframe.to_csv(csv_filepath, mode='a', header=not os.path.exists(csv_filepath))
    
  wandb.finish()


if __name__ =="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default=None, help='Filepath to Doom agent model checkpoint file to load')
  parser.add_argument('--seed', type=int, default=42, help='Random seed')
  parser.add_argument('--episode_max_steps', type=int, default=25000, help='Maximum steps/ticks before an episode ends')
  parser.add_argument('--device', default='cuda', help='CUDA or CPU')
  parser.add_argument('--epsilon', type=float, default=1.0, help="Starting epislon value (probability of taking random actions) for the epsilon-greedy policy in training.")
  parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
  parser.add_argument('--seq_len', type=int, default=30, help='Sequence Length (chunk length)')
  parser.add_argument('--explore_steps', type=int, default=5000, help='Exploration steps to take before training')
  parser.add_argument('--save_steps', type=int, default=50000, help='Interval of steps to take before saving the models')
  parser.add_argument('--num_episodes', type=int, default=10000, help="Number of episodes")
  parser.add_argument("--map", type=str, default="E1M1", help="The doom map name to play/train on")
  
  args = parser.parse_args()
  
  main(args)
  
  