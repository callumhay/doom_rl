#ifndef __DOOMGUY_HPP__
#define __DOOMGUY_HPP__

#include <string>
#include <vector>
#include <array>

#include <torch/torch.h>

#include "DoomEnv.hpp"
#include "CosAnnealLRScheduler.hpp"
#include "DoomGuyNet.hpp"

class ReplayMemory;

class DoomGuy {
public:
  DoomGuy(const std::string& saveDir, size_t stepsPerEpisode, 
    size_t stepsExplore, size_t stepsBetweenSaves, size_t stepsBetweenSyncs, 
    double startEpsilon, double epsilonMin, double epsilonDecay, double learningRate);


  void train(bool on = true) { this->net->train(on); }

  // Given a state, choose an epsilon-greedy action
  DoomEnv::Action act(torch::Tensor state, std::unique_ptr<DoomEnv>& env); 

  // Update online action value (Q) function with a batch of experiences
  // Returns <mean_q, loss>
  std::tuple<double, double> learn(std::unique_ptr<ReplayMemory>& replayMemory, size_t batchSize);

  void episodeEnded(size_t episodeNum);

  auto getNetworkVersion() const { return this->net->getCurrVersion(); }
  auto getCurrStep() const { return this->currStep; }
  auto getEpsilon() const { return this->epsilon; }
  auto getLearningRate() const { return this->optimizer.param_groups()[0].options().get_lr(); }

  std::string optimizerSaveFilepath(size_t version, size_t saveNum);
  std::string netSaveFilepath(size_t version, size_t saveNum);
  void save();
  void load(const std::string& chkptFilepath);

private:
  std::string saveDir;
  size_t currStep;
  size_t stepsPerEpisode;
  size_t stepsBetweenSaves;    // Number of steps between saving the networks to disk
  size_t stepsBetweenSyncs;    // Number of steps between synchronizing between the Q-Target and Online networks
  size_t stepsExplore;         // Number of steps to explore before training starts
  size_t stepsBetweenLearning; // Number of steps between when we actually learn (generally it's good not to learn every step with DQN)

  bool useCuda;      // Whether or not the model makes use of CUDA
  size_t saveNumber; // Keep track of the number of saves so far

  // Training options
  double epsilon;                 // The current exploration probability in the greedy-epsilon policy
  double epsilonDecayMultiplier;  // epislon decays as a multiple of this value after each step
  double epsilonMin;              // The minimum allowable epislon value
  double gamma;                   // The discount factor
  
  DoomGuyNet net;
  torch::optim::Adam optimizer;
  std::unique_ptr<CosAnnealLRScheduler> lrScheduler;
  torch::nn::SmoothL1Loss lossFn;

  void rebuildScheduler() {
    // NOTE: an epoch is one complete pass through the training data... in RL this is pretty meaningless,
    // to start so that it doesn't overcompensate at the start
    auto expStepsPerEpoch = this->stepsPerEpisode/100;
    if (this->lrScheduler != nullptr) {
      expStepsPerEpoch = this->lrScheduler->getAvgBatchesPerEpoch();
    }
    this->lrScheduler = std::make_unique<CosAnnealLRScheduler>(this->optimizer, expStepsPerEpoch);
  }

  torch::Tensor tdEstimate(torch::Tensor stateBatch, torch::Tensor actionBatch);
  torch::Tensor tdTarget(torch::Tensor rewardBatch, torch::Tensor nextStateBatch, torch::Tensor doneBatch);
  torch::Scalar updateQOnline(torch::Tensor tdEstimate, torch::Tensor tdTarget);
};

#endif // __DOOMGUY_HPP__