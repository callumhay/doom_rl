#include <assert.h>
#include <cstdlib>
#include <algorithm>
#include <random>

#include "DoomGuy.hpp"

using State  = DoomEnv::State;
using Action = DoomEnv::Action;

constexpr size_t DEFAULT_REPLAY_BATCH_SIZE = 32;
constexpr size_t REPLAY_MEMORY_MAX_SIZE    = 10000;

DoomGuy::DoomGuy(const std::string& saveDir) : 
saveDir(saveDir), currStep(0), batchSize(DEFAULT_REPLAY_BATCH_SIZE), 
numStepsBetweenSaves(5e5), useCuda(torch::cuda::is_available()), 
net(std::make_unique<DoomGuyNet>(
    torch::tensor({State::NUM_CHANNELS, State::TENSOR_INPUT_HEIGHT, State::TENSOR_INPUT_WIDTH}),
    torch::tensor({Action::numActions})))
 {

  this->replayMemory.reserve(REPLAY_MEMORY_MAX_SIZE);

  if (this->useCuda) {
    this->net->to(torch::device(torch::kCUDA));
  }

  // TODO: put in options object?
  this->explorationRate = 1.0;
  this->explorationRateDecay = 0.99999975;
  this->explorationRateMin = 0.1;
}

/**
 * Given a state, choose an epsilon-greedy action and update value of step.
 */
DoomEnv::Action DoomGuy::act(DoomEnv::StatePtr state) {
  auto random = static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);

  DoomEnv::Action action;
  if (random < this->explorationRate) {
    // Explore
    action = static_cast<DoomEnv::Action>(std::rand() % DoomEnv::numActions);
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

void DoomGuy::cache(DoomEnv::StatePtr& state, DoomEnv::StatePtr& nextState, DoomEnv::Action action, double reward, bool done) {
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
  assert(this->replayMemory.size() >= n);

  std::vector<DoomGuy::ReplayData> samples;
  samples.reserve(n);
  std::sample(
    this->replayMemory.begin(), this->replayMemory.end(), 
    std::back_inserter(samples), n, std::mt19937{std::random_device{}()}
  );

  return samples;
}