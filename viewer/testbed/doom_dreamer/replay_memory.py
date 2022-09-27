from typing import Tuple

import torch
import numpy as np 


class ReplayMemory():
  def __init__(self,
      capacity: int,
      observation_shape: Tuple[int],
      action_size: int,
      seq_len: int, 
      batch_size: int,
  ):
    self.capacity = capacity
    self.observation_shape = observation_shape
    self.action_size = action_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    self.idx = 0
    self.full = False
    self.observation = np.empty((capacity, *observation_shape), dtype=np.float32) 
    self.action      = np.empty((capacity, action_size), dtype=np.float32)
    self.reward      = np.empty((capacity,), dtype=np.float32) 
    self.terminal    = np.empty((capacity,), dtype=bool)

  def add(self, observation: np.ndarray, action: np.ndarray, reward: float, done: bool):
    self.observation[self.idx] = observation
    self.action[self.idx] = action
    #self.action[self.idx].fill(0)
    #self.action[self.idx][action] = 1 
    self.reward[self.idx] = reward
    self.terminal[self.idx] = done
    self.idx = (self.idx + 1) % self.capacity
    self.full = self.full or self.idx == 0

  def sample(self):
    n = self.batch_size
    l = self.seq_len+1
    observation, action, reward, terminal = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
    observation, action, reward, terminal = self._shift_sequences(observation, action, reward, terminal)
    return observation, action, reward, terminal
  
  
  def _sample_idx(self, seq_len: int):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.capacity if self.full else self.idx - seq_len)
      idxs = np.arange(idx, idx + seq_len) % self.capacity
      valid_idx = not self.idx in idxs[1:] 
    return idxs

  def _retrieve_batch(self, idxs: np.ndarray, batch_size: int, seq_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vec_idxs = idxs.transpose().reshape(-1)
    observation = self.observation[vec_idxs]
    return (
      observation.reshape(seq_len, batch_size, *self.observation_shape), 
      self.action[vec_idxs].reshape(seq_len, batch_size, -1), 
      self.reward[vec_idxs].reshape(seq_len, batch_size), 
      self.terminal[vec_idxs].reshape(seq_len, batch_size)
    )
  
  def _shift_sequences(self, observations, actions, rewards, terminals):
    observations = observations[1:]
    actions = actions[:-1]
    rewards = rewards[:-1]
    terminals = terminals[:-1]
    return observations, actions, rewards, terminals