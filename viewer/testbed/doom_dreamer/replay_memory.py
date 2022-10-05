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
    self.position    = np.empty((capacity, 3), dtype=np.float32)
    self.heading     = np.empty((capacity,), dtype=np.float32)
    self.reward      = np.empty((capacity,), dtype=np.float32) 
    self.terminal    = np.empty((capacity,), dtype=bool)

  def add(self, observation: np.ndarray, action: np.ndarray, position: np.ndarray, heading: float, reward: float, done: bool):
    self.observation[self.idx] = observation
    self.action[self.idx] = action
    self.position[self.idx] = position
    self.heading[self.idx] = heading
    self.reward[self.idx] = reward
    self.terminal[self.idx] = done
    self.idx = (self.idx + 1) % self.capacity
    self.full = self.full or self.idx == 0

  def sample(self):
    n = self.batch_size
    l = self.seq_len+1
    observation, action, position, heading, reward, terminal = self._retrieve_batch(np.asarray([self._sample_idx(l) for _ in range(n)]), n, l)
    observation, action, position, heading, reward, terminal = self._shift_sequences(observation, action, position, heading, reward, terminal)
    return observation, action, position, heading, reward, terminal
  
  
  def _sample_idx(self, seq_len: int):
    valid_idx = False
    while not valid_idx:
      idx = np.random.randint(0, self.capacity if self.full else self.idx - seq_len)
      idxs = np.arange(idx, idx + seq_len) % self.capacity
      valid_idx = not self.idx in idxs[1:] 
    return idxs

  def _retrieve_batch(self, idxs: np.ndarray, batch_size: int, seq_len: int):
    vec_idxs = idxs.transpose().reshape(-1)
    observation = self.observation[vec_idxs]
    return (
      observation.reshape(seq_len, batch_size, *self.observation_shape), 
      self.action[vec_idxs].reshape(seq_len, batch_size, -1), 
      self.position[vec_idxs].reshape(seq_len, batch_size, -1), 
      self.heading[vec_idxs].reshape(seq_len, batch_size), 
      self.reward[vec_idxs].reshape(seq_len, batch_size), 
      self.terminal[vec_idxs].reshape(seq_len, batch_size)
    )
  
  def _shift_sequences(self, observations, actions, positions, headings, rewards, terminals):
    return observations[1:], actions[:-1], positions[1:], headings[1:], rewards[:-1], terminals[:-1]