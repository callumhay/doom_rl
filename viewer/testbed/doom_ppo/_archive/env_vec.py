from typing import List
import numpy as np
from doom_env import NUM_LABEL_CLASSES, DoomEnv


class EnvVec(object):
  def __init__(self, envs: List[DoomEnv]):
    self.envs = envs
    
  def close(self):
    for env in self.envs:
      env.close()
  
  def action_space_size(self):
    return self.envs[0].action_size
  
  def observation_shape(self):
    return self.envs[0].observation_shape()
  
  def label_shape(self):
    return (NUM_LABEL_CLASSES,)
  
  def reset(self):
    obs = []
    labels = []
    for env in self.envs:
      ob, label = env.reset()
      obs.append(ob)
      labels.append(label)
    return np.stack(obs), np.stack(labels)
  
  def step(self, actions):
    obs = []
    labels = []
    rewards = []
    dones = []
    infos = []
    for i, env in enumerate(self.envs):
      ob, label, reward, done, info = env.step(actions[i])
      obs.append(ob)
      labels.append(label)
      rewards.append(reward)
      dones.append(done)
      infos.append(info)
    return np.stack(obs), np.stack(labels), rewards, dones, infos
  
  