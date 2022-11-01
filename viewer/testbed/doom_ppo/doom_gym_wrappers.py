
import gym
import numpy as np
import torch
from torchvision import transforms as T
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn

class DoomObservation(gym.ObservationWrapper):
  def __init__(self, env, shape) -> None:
    super().__init__(env)
    self.shape = shape
    obs_shape = (self.observation_space[0].shape[-1],) + self.shape
    obs_space_list = [
      gym.spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32), 
      self.observation_space[1]
    ]
    self.observation_space = gym.spaces.Tuple(obs_space_list)

  def permute_observation(self, observation):
    observation = np.transpose(observation, (2,0,1))
    observation = torch.tensor(observation.copy(), dtype=torch.float)
    return observation
  
  def observation(self, observation):
    obs = self.permute_observation(observation[0])
    obs = T.Resize(self.shape, T.InterpolationMode.NEAREST)(obs).squeeze(0) / 255.0
    observation[0] = obs.numpy()
    return observation


class DoomMaxAndSkipEnv(gym.Wrapper):
  """
  Return only every ``skip``-th frame (frameskipping)

  :param env: the environment
  :param skip: number of ``skip``-th frame
  """

  def __init__(self, env: gym.Env, skip: int = 4):
    gym.Wrapper.__init__(self, env)
    # most recent raw observations (for max pooling across time steps)
    self._obs_buffer = [None, None]
    self._tuple_obs = True
    self._skip = skip

  def step(self, action: int) -> GymStepReturn:
    """
    Step the environment with the given action
    Repeat action, sum reward, and max over last observations.

    :param action: the action
    :return: observation, reward, done, information
    """
    total_reward = 0.0
    done = None
  
    for i in range(self._skip):
      obs, reward, done, info = self.env.step(action)
      if i == self._skip - 2:
        self._obs_buffer[0] = obs
      if i == self._skip - 1:
        self._obs_buffer[1] = obs
      total_reward += reward
      if done:
          break

    max_frame = [[] for _ in range(len(obs))]
    for i, obs in enumerate(self._obs_buffer):
      for j,obs_part in enumerate(obs):
        max_frame[j].append(obs_part)
        
    for i, obs_parts in enumerate(max_frame):
      max_frame[i] = np.max(obs_parts, axis=0)

    return max_frame, total_reward, done, info

  def reset(self, **kwargs) -> GymObs:
    return self.env.reset(**kwargs)