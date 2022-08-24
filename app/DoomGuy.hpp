#ifndef __DOOMGUY_HPP__
#define __DOOMGUY_HPP__

#include <string>
#include <vector>
#include <array>

#include <torch/torch.h>

#include "DoomEnv.hpp"
#include "CosAnnealLRScheduler.hpp"

class DoomGuyNet;

class DoomGuy {
public:
  DoomGuy(const std::string& saveDir, size_t stepsPerEpisode, size_t stepsExplore, size_t stepsBetweenSaves, size_t stepsBetweenSyncs, 
    double startEpsilon, double epsilonDecay, double learningRate);

  // Given a state, choose an epsilon-greedy action
  DoomEnv::Action act(DoomEnv::StatePtr& state); 

  // Add the experience to memory
  void cache(DoomEnv::StatePtr& state, DoomEnv::StatePtr& nextState, DoomEnv::Action action, double reward, bool done);

  // Update online action value (Q) function with a batch of experiences
  // Returns <mean_q, loss>
  std::tuple<double, double> learn();

  void episodeEnded(size_t episodeNum);

  auto getCurrStep() const { return this->currStep; }
  auto getEpsilon() const { return this->epsilon; }
  auto getLearningRate() const { return static_cast<torch::optim::AdamOptions&>(this->optimizer->param_groups()[0].options()).lr(); }

  void save();
  void load(const std::string& chkptFilepath);

private:
  std::string saveDir;
  size_t currStep;
  size_t stepsPerEpisode;
  size_t stepsBetweenSaves;  // Number of steps between saving the networks to disk
  size_t stepsBetweenSyncs;  // Number of steps between synchronizing between the Q-Target and Online networks
  size_t stepsExplore;       // Number of steps to explore before training starts

  bool useCuda;      // Whether or not the model makes use of CUDA
  size_t saveNumber; // Keep track of the number of saves so far

  // Training options
  size_t batchSize;               // Batch size when training with replay memory
  double epsilon;                 // The current exploration probability in the greedy-epsilon policy
  double epsilonDecayMultiplier;  // epislon decays as a multiple of this value after each step
  double epsilonMin;              // The minimum allowable epislon value
  double gamma;                   // The discount factor
  
  std::shared_ptr<DoomGuyNet> net; // NOTE: This needs to be shared in order for torch::save to work
  std::unique_ptr<torch::optim::Adam> optimizer;
  std::unique_ptr<CosAnnealLRScheduler> lrScheduler;
  torch::nn::SmoothL1Loss lossFn;

  // Data that is stored in the replay memory - an array of tensors corresponding to
  // [state, nextState, action, reward, done]
  using ReplayData = std::array<torch::Tensor, 5>;
  std::vector<ReplayData> replayMemory;

  void rebuildOptimizer(double lr);

  ReplayData recall();
  std::vector<ReplayData> randomSamples(size_t n);
  
  torch::Tensor tdEstimate(torch::Tensor stateBatch, torch::Tensor actionBatch);
  torch::Tensor tdTarget(torch::Tensor rewardBatch, torch::Tensor nextStateBatch, torch::Tensor doneBatch);
  torch::Scalar updateQOnline(torch::Tensor tdEstimate, torch::Tensor tdTarget);
};

#endif // __DOOMGUY_HPP__