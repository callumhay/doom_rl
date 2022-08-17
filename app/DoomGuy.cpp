#include <assert.h>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <iostream>
#include <sstream>

#include "DoomGuy.hpp"
#include "DoomGuyNet.hpp"

using State  = DoomEnv::State;
using Action = DoomEnv::Action;

constexpr size_t DEFAULT_REPLAY_BATCH_SIZE = 32;
constexpr size_t REPLAY_MEMORY_MAX_SIZE    = 10000;

DoomGuy::DoomGuy(const std::string& saveDir) : 
saveDir(saveDir), currStep(0), batchSize(DEFAULT_REPLAY_BATCH_SIZE), 
stepsBetweenSaves(5e5), stepsBetweenSyncs(1e4), stepsExplore(1e4),
useCuda(torch::cuda::is_available()), gamma(0.9), lossFn(),
net(std::make_shared<DoomGuyNet>(
    torch::tensor({static_cast<int>(State::NUM_CHANNELS), static_cast<int>(State::TENSOR_INPUT_HEIGHT), static_cast<int>(State::TENSOR_INPUT_WIDTH)}), DoomEnv::numActions
)) {
  assert(this->stepsExplore >= DEFAULT_REPLAY_BATCH_SIZE);

  this->replayMemory.reserve(REPLAY_MEMORY_MAX_SIZE);

  if (this->useCuda) {
    this->net->to(torch::kCUDA);
  }
  
  this->optimizer = std::make_unique<torch::optim::Adam>(this->net->parameters(), torch::optim::AdamOptions(0.00025));

  // TODO: put in options object?
  this->explorationRate = 1.0;
  this->explorationRateDecay = 0.99999975;
  this->explorationRateMin = 0.1;
}

/**
 * Given a state, choose an epsilon-greedy action and update value of step.
 */
Action DoomGuy::act(DoomEnv::StatePtr& state) {
  auto random = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);

  Action action;
  if (random < this->explorationRate) {
    // Explore
    action = static_cast<Action>(std::rand() % DoomEnv::numActions);
  }
  else {
    // Exploit
    auto stateTensor = this->useCuda ? state->tensor().cuda() : state->tensor();
    stateTensor.unsqueeze(0);
    //auto actionOutTensor = this->net(stateTensor, DoomGuyNet::Model::Online);
    //action = static_cast<DoomEnv::Action>(torch::argmax(actionOutTensor, 1).item());
  }

  // Decrease the exporationRate
  this->explorationRate *= this->explorationRateDecay;
  this->explorationRate = std::max(this->explorationRateMin, this->explorationRate);

  // Increment the step
  this->currStep++;

  return action;
}

void DoomGuy::cache(DoomEnv::StatePtr& state, DoomEnv::StatePtr& nextState, Action action, double reward, bool done) {
  // Treat the replay buffer as a circular buffer, removing the oldest sample if we go over the max size
  if (this->replayMemory.size() == REPLAY_MEMORY_MAX_SIZE) {
    std::swap(this->replayMemory[0], this->replayMemory.back());
    this->replayMemory.pop_back();
  }

  if (this->useCuda) {
    this->replayMemory.push_back({
      state->tensor().cuda(), 
      nextState->tensor().cuda(),
      torch::tensor({static_cast<int>(action)}).cuda(), 
      torch::tensor({reward}).cuda(),
      torch::tensor({done}).cuda(),
    });
  }
  else {
    this->replayMemory.push_back({
      state->tensor(), 
      nextState->tensor(),
      torch::tensor({static_cast<int>(action)}), 
      torch::tensor({reward}),
      torch::tensor({done}),
    });
  }
}

std::tuple<double, double> DoomGuy::learn() {
  if (this->currStep % this->stepsBetweenSyncs == 0) {
    this->net->syncTarget();
  }
  if (this->currStep % this->stepsBetweenSaves == 0) {
    this->save();
  }
  if (this->currStep < this->stepsExplore) {
    return std::make_tuple(-1.0,-1.0);
  }
  // NOTE: We currently learn at every step, so no need for this.
  //if (this->currStep % this->stepsBetweenLearningOnline != 0) { return std::make_tuple(-1.0,-1.0); }

  // Sample from memory
  auto [stateBatch, nextStateBatch, actionBatch, rewardBatch, doneBatch] = this->recall();

  // Get the Temporal Difference (TD) estimate and target
  auto tdEst = this->tdEstimate(stateBatch, actionBatch);
  auto tdTgt = this->tdTarget(rewardBatch, nextStateBatch, doneBatch);

  // Backprogogate the loss through the Q-Online Network
  auto loss = this->updateQOnline(tdEst, tdTgt);

  return std::make_tuple(tdEst.mean().item<double>(), loss.toDouble());
}

/**
 * Recall a tuple of stacked tensors, each stack is the size of the batch
 * and is setup to be fed directly to the NN.
 */
DoomGuy::ReplayData DoomGuy::recall() {
  auto samples = this->randomSamples(DEFAULT_REPLAY_BATCH_SIZE);
  assert(samples.size() == DEFAULT_REPLAY_BATCH_SIZE);

  // Stack the samples...
  std::array<
    std::array<torch::Tensor, DEFAULT_REPLAY_BATCH_SIZE>, 
    std::tuple_size<DoomGuy::ReplayData>::value
  > stackLists;

  for (auto i = 0; i < stackLists.size(); i++) {
    auto& stackList = stackLists[i];
    for (auto j = 0; j < stackList.size(); j++) {
      stackList[j] = std::move(samples[j][i]);
    }
  }

  ReplayData result;
  for (auto i = 0; i < result.size(); i++) {
    result[i] = torch::stack(stackLists[i]);
  }
  return result;
}

// NOTE: Move semantics keep the vector from being copied!
std::vector<DoomGuy::ReplayData> DoomGuy::randomSamples(size_t n) {
  assert(this->replayMemory.size() >= n && "Not enough replay memory has been stored yet, increase exploration time and/or decrease your batch size.");

  std::vector<DoomGuy::ReplayData> samples;
  samples.reserve(n);
  std::sample(
    this->replayMemory.begin(), this->replayMemory.end(), 
    std::back_inserter(samples), n, std::mt19937{std::random_device{}()}
  );

  return samples;
}

torch::Tensor DoomGuy::tdEstimate(torch::Tensor stateBatch, torch::Tensor actionBatch) {
  using namespace torch::indexing;

  // Get the network output, a tensor of the form [batch_size, Q(s,a)]
  // Where Q(s,a) is the online network Q-Values for each action, with a dimension of DoomEnv::numActions
  auto currentQ = this->net->forward(stateBatch, DoomGuyNet::Model::Online).index({Slice(), actionBatch});
  return currentQ;
}

torch::Tensor DoomGuy::tdTarget(torch::Tensor rewardBatch, torch::Tensor nextStateBatch, torch::Tensor doneBatch) {
  using namespace torch::indexing;
  torch::NoGradGuard no_grad;

  auto nextStateQ = this->net->forward(nextStateBatch, DoomGuyNet::Model::Online);
  auto bestAction = torch::argmax(nextStateQ, 1);
  auto nextQ = this->net->forward(nextStateBatch, DoomGuyNet::Model::Target).index({Slice(), bestAction});

  return (rewardBatch + (1.0 - doneBatch.to(torch::kFloat)) * this->gamma * nextQ).to(torch::kFloat);
}

torch::Scalar DoomGuy::updateQOnline(torch::Tensor tdEstimate, torch::Tensor tdTarget) {
  auto loss = this->lossFn(tdEstimate, tdTarget);
  this->optimizer->zero_grad();
  loss.backward();
  this->optimizer->step();
  return loss.item();
}

void DoomGuy::save() {
  std::stringstream savePath;
  savePath << this->saveDir << "/doomguy_net_" << static_cast<int>(this->currStep/this->stepsBetweenSaves) << ".chkpt";
  torch::save(this->net, savePath.str());
  std::cout << "DoomGuyNet saved to " << savePath.str() << " at step " << this->currStep << std::endl;
}