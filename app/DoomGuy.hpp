#ifndef __DOOMGUY_HPP__
#define __DOOMGUY_HPP__

#include <string>
#include <vector>
#include <array>

#include <torch/torch.h>

#include "DoomEnv.hpp"

class DoomGuyNet;

class DoomGuy {
public:
  DoomGuy(const std::string& saveDir);

  // Given a state, choose an epsilon-greedy action
  DoomEnv::Action act(DoomEnv::StatePtr state); 

  // Add the experience to memory
  void cache(DoomEnv::StatePtr& state, DoomEnv::StatePtr& nextState, DoomEnv::Action action, double reward, bool done);

  // Update online action value (Q) function with a batch of experiences
  //void learn();

private:
  std::string saveDir;
  size_t currStep;
  size_t numStepsBetweenSaves;

  bool useCuda;

  // Training options
  size_t batchSize;             // Batch size when training with replay memory
  double explorationRate;       // 'epsilon' in the greedy-epsilon policy
  double explorationRateDecay;  // The rate with which epislon decays as the episode plays
  double explorationRateMin;    // The minimum epislon value
  
  std::unique_ptr<DoomGuyNet> net;

  // Data that is stored in the replay memory - an array of tensors corresponding to
  // [state, nextState, action, reward, done]
  using ReplayData = std::array<torch::Tensor, 5>;
  std::vector<ReplayData> replayMemory;

  ReplayData recall();
  std::vector<ReplayData> randomSamples(size_t n);
  

};

#endif // __DOOMGUY_HPP__