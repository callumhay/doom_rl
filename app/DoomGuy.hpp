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
  DoomGuy(const std::string& saveDir, size_t stepsExplore=1e3);

  // Given a state, choose an epsilon-greedy action
  DoomEnv::Action act(DoomEnv::StatePtr& state); 

  // Add the experience to memory
  void cache(DoomEnv::StatePtr& state, DoomEnv::StatePtr& nextState, DoomEnv::Action action, double reward, bool done);

  // Update online action value (Q) function with a batch of experiences
  // Returns <mean_q, loss>
  std::tuple<double, double> learn();

  auto getCurrStep() const { return this->currStep; }
  auto getExplorationRate() const { return this->explorationRate; }

private:
  std::string saveDir;
  size_t currStep;
  size_t stepsBetweenSaves;  // Number of steps between saving the networks to disk
  size_t stepsBetweenSyncs;  // Number of steps between synchronizing between the Q-Target and Online networks
  size_t stepsExplore;       // Number of steps to explore before training starts

  bool useCuda;

  // Training options
  size_t batchSize;             // Batch size when training with replay memory
  double explorationRate;       // 'epsilon' in the greedy-epsilon policy
  double explorationRateDecay;  // The rate with which epislon decays as the episode plays
  double explorationRateMin;    // The minimum epislon value
  double gamma;                 // The discount factor
  
  std::shared_ptr<DoomGuyNet> net; // NOTE: This needs to be shared in order for torch::save to work
  std::unique_ptr<torch::optim::Adam> optimizer;
  torch::nn::SmoothL1Loss lossFn;

  // Data that is stored in the replay memory - an array of tensors corresponding to
  // [state, nextState, action, reward, done]
  using ReplayData = std::array<torch::Tensor, 5>;
  std::vector<ReplayData> replayMemory;

  ReplayData recall();
  std::vector<ReplayData> randomSamples(size_t n);
  
  torch::Tensor tdEstimate(torch::Tensor stateBatch, torch::Tensor actionBatch);
  torch::Tensor tdTarget(torch::Tensor rewardBatch, torch::Tensor nextStateBatch, torch::Tensor doneBatch);
  torch::Scalar updateQOnline(torch::Tensor tdEstimate, torch::Tensor tdTarget);

  void syncQTarget();

  void save();

};

#endif // __DOOMGUY_HPP__