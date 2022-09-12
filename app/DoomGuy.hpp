#ifndef __DOOMGUY_HPP__
#define __DOOMGUY_HPP__

#include <assert.h>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <cmath>

#include <torch/torch.h>

#include "DoomEnv.hpp"
#include "DoomGuyNet.hpp"

class ReplayMemory;
class DoomRLCmdOpts;
class LearningRateScheduler;

class DoomGuy {
public:
  DoomGuy(const std::string& saveDir, const std::unique_ptr<DoomRLCmdOpts>& cmdOpts);

  void train(bool on = true) { this->net->train(on); }

  void episodeEnded(size_t episodeNum);

  auto getNetworkVersion() const { return this->net->getCurrVersion(); }
  auto getCurrStep() const { return this->currStep; }
  auto getEpsilon() const { return this->epsilon; }
  auto getLearningRate() const {
    auto optLR = this->optimizer.param_groups()[0].options().get_lr();
    //assert(std::abs(this->lrScheduler->getCurrentLR()-optLR) < 1e-6);
    return optLR;
  }

  std::string optimizerSaveFilepath(size_t version, size_t minorVersion, size_t saveNum);
  std::string netSaveFilepath(size_t version, size_t minorVersion, size_t saveNum);
  std::string replayMemSaveFilepath() const;
  void save(const ReplayMemory& replayMemory);
  void load(const std::string& chkptFilepath);
  void load(const std::string& chkptFilepath, ReplayMemory& replayMemory);

  // Given a state, choose an epsilon-greedy action
  DoomEnv::Action act(torch::Tensor state, std::unique_ptr<DoomEnv>& env); 

  // Update online action value (Q) function with a batch of experiences
  // Returns <mean_q, loss>
  std::tuple<double, double> learn(std::unique_ptr<DoomEnv>& env, std::unique_ptr<ReplayMemory>& replayMemory, size_t batchSize);


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
  std::unique_ptr<LearningRateScheduler> lrScheduler;
  torch::nn::SmoothL1Loss lossFn;

  std::tuple<double, double> learnRandom(std::unique_ptr<ReplayMemory>& replayMemory, size_t batchSize);
  std::tuple<double, double> learnSequence(std::unique_ptr<DoomEnv>& env, std::unique_ptr<ReplayMemory>& replayMemory, size_t batchSize);



  torch::Tensor tdEstimate(torch::Tensor stateBatch, torch::Tensor actionBatch);
  torch::Tensor tdTarget(torch::Tensor rewardBatch, torch::Tensor nextStateBatch, torch::Tensor doneBatch);
  torch::Scalar updateQOnline(torch::Tensor tdEstimate, torch::Tensor tdTarget);
};

#endif // __DOOMGUY_HPP__