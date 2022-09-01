#ifndef __REPLAYMEMORY_HPP__
#define __REPLAYMEMORY_HPP__

#include <cassert>
#include <algorithm>
#include <vector>
#include <array>

#include <torch/torch.h>

#include "utils/RNG.hpp"

class ReplayMemory {
public:
  // https://acsweb.ucsd.edu/~wfedus/pdf/replay.pdf
  static constexpr size_t DEFAULT_REPLAY_BATCH_SIZE = 32;
  static constexpr size_t REPLAY_MEMORY_MAX_SIZE = 72000;

  ReplayMemory(size_t stateWidth, size_t stateHeight, size_t stateChannels) {
    auto numData = stateWidth*stateHeight*stateChannels+4;
    this->capacity = std::max<size_t>(
      REPLAY_MEMORY_MAX_SIZE, 
      (REPLAY_MEMORY_FP_RATIO_FPS/numData) * REPLAY_MEMORY_FP_RATIO_SIZE
    );
    std::cout << "Replay memory capacity set to " << this->capacity << std::endl;

    this->states.reserve(this->capacity);
    this->actions.reserve(this->capacity);
    this->rewards.reserve(this->capacity);
    this->dones.reserve(this->capacity);
  }
  ReplayMemory(const ReplayMemory&) = delete;
  ReplayMemory& operator=(const ReplayMemory&) = delete;
  ~ReplayMemory() = default;

  size_t getCacheSize() { return this->dones.size(); }

  void initState(torch::Tensor initState) { this->appendToReplayVec(initState, this->states); }

  void cache(torch::Tensor nextState, int action, double reward, bool done) {
    if (!done) {
      this->appendToReplayVec(nextState, this->states);
    }
    this->appendToReplayVec(torch::tensor({action}), this->actions);
    this->appendToReplayVec(torch::tensor({reward}), this->rewards);
    this->appendToReplayVec(torch::tensor({done ? 1.0 : 0.0}), this->dones);
  }

  using ReplayData = std::array<torch::Tensor, 5>;
  using SequenceReplayData = std::array<torch::Tensor, 5/*7*/>;

  // Produces an array of 5 tensors, each tensor with a shape of [batchSize, ...feature_values]
  // The 5 tensors correspond to, in order: [stateBatch, nextStateBatch, actionBatch, rewardBatch, doneBatch]
  ReplayData randomRecall(size_t batchSize) { return this->randomSamples(batchSize); }

  // Produces an array of 5 specialized tensors:
  // [stateSeq, nextStateSeq, actionBatch, rewardBatch, doneBatch]
  // The first 2 tensors are formatted such that they capture a sequence of length 'sequenceLen'
  // The last 3 tensors are formatted similar to the randomRecall function, capturing the rewards and dones
  // for the state corresponding to the end of the sequence (last state) of the stateSeq tensor
  SequenceReplayData sequenceRecall(size_t batchSize, size_t sequenceLen);


private:
  using TensorVec = std::vector<torch::Tensor>;
  
  // If the capacity of the replay memory is too big we WILL run out of memory !!
  // For 320x200x3 floating point value state size, the max number for ~65GB is 72000
  static constexpr double REPLAY_MEMORY_FP_RATIO_FPS  = (320*200*3+4);
  static constexpr double REPLAY_MEMORY_FP_RATIO_SIZE = 41200;

  size_t capacity;

  TensorVec states;
  TensorVec actions;
  TensorVec rewards;
  TensorVec dones;

  void appendToReplayVec(torch::Tensor data, TensorVec& vec) {
    // Remove oldest sample if we go over the max size
    if (vec.size() >= this->capacity) {
      std::swap(vec[0], vec.back());
      vec.pop_back();
    }
    vec.push_back(data);
    assert(vec.capacity() == this->capacity);
  }

  // Get a vector of random indicies where each element in the vector is an array of size 2, the first element
  // is the random index and the next is that same index + 1. The choice of indices is limitted by the given
  // sequenceLen (i.e., the choices will be made far enough back in the history of the replay memory to support
  // the given sequenceLen). 
  auto getRandomIndices(size_t batchSize, size_t sequenceLen=1) {
    assert(
      this->states.size() >= batchSize && this->actions.size() >= batchSize && 
      this->rewards.size() >= batchSize && this->dones.size() >= batchSize
    );

    auto rndIndices = RNG::getInstance()->genShuffledIndices(batchSize, this->dones.size()-sequenceLen);
    auto rndIndicesPlus1 = std::vector<size_t>();
    rndIndicesPlus1.reserve(rndIndices.size());
    std::transform(rndIndices.cbegin(), rndIndices.cend(), std::back_inserter(rndIndicesPlus1),
      [](size_t elem) { return elem+1; }
    );

    return std::array<std::vector<size_t>,2>({rndIndices, rndIndicesPlus1});
  }

  // Produces an array of [state, nextState, actions, rewards, dones], where each element is a torch::Tensor
  // of the specified batch size
  ReplayData randomSamples(size_t batchSize) {

    auto [rndIndices, rndIndicesPlus1] = this->getRandomIndices(batchSize);
    auto stateSamples     = ReplayMemory::sampleFromVec(rndIndices, this->states);
    auto nextStateSamples = ReplayMemory::sampleFromVec(rndIndicesPlus1, this->states);
    auto actionSamples    = ReplayMemory::sampleFromVec(rndIndices, this->actions);
    auto rewardSamples    = ReplayMemory::sampleFromVec(rndIndices, this->rewards);
    auto doneSamples      = ReplayMemory::sampleFromVec(rndIndices, this->dones);

    return {
      torch::stack(stateSamples),
      torch::stack(nextStateSamples),
      torch::stack(actionSamples),
      torch::stack(rewardSamples),
      torch::stack(doneSamples)
    };
  };

  static TensorVec sampleFromVec(const std::vector<size_t>& randomIndices, TensorVec& sampleVec) {
    TensorVec samples;
    samples.reserve(randomIndices.size());
    for (auto idx : randomIndices) {
      samples.push_back(sampleVec[idx]);
    }
    return samples;
  }

  static TensorVec sequenceFromVec(size_t startIdx, size_t sequenceLen, TensorVec& sampleVec) {
    TensorVec seqSamples;
    seqSamples.reserve(sequenceLen);
    for (auto i = startIdx; i < (startIdx+sequenceLen); i++) {
      seqSamples.push_back(sampleVec[i]);
    }
    return seqSamples;
  }

};

inline ReplayMemory::SequenceReplayData ReplayMemory::sequenceRecall(size_t batchSize, size_t sequenceLen) {
  assert(
    this->states.size() >= batchSize && this->actions.size() >= batchSize && 
    this->rewards.size() >= batchSize && this->dones.size() >= batchSize
  );  
  auto rndStartIndices = RNG::getInstance()->genShuffledIndices(batchSize, this->dones.size()-sequenceLen);
  
  TensorVec stateSeqBatch; stateSeqBatch.reserve(batchSize);
  //TensorVec nextStateSeqBatch; nextStateSeqBatch.reserve(batchSize);
  //TensorVec actionSeqBatch; actionSeqBatch.reserve(batchSize);
  //TensorVec nextActionSeqBatch; nextActionSeqBatch.reserve(batchSize);

  // Build the sequences and as we go add them to the batches
  for (auto batchIdx = 0; batchIdx < batchSize; batchIdx++) {
    auto rndStartIdx = rndStartIndices[batchIdx];
    //auto rndStartIdxPlus1 = rndStartIdx+1;

    // Generate a sequence vector for each piece of data
    auto stateSeqSamples = ReplayMemory::sequenceFromVec(rndStartIdx, sequenceLen, this->states);
    //auto nextStateSeqSamples  = ReplayMemory::sequenceFromVec(rndStartIdxPlus1, sequenceLen, this->states);
    //auto actionSeqSamples     = ReplayMemory::sequenceFromVec(rndStartIdx,      sequenceLen, this->actions);
    //auto nextActionSeqSamples = ReplayMemory::sequenceFromVec(rndStartIdxPlus1, sequenceLen, this->actions);

    // We need to satisfy the following tensor shape for each final batch of sequences:
    // [batch_size, sequence_length, features]
    stateSeqBatch.push_back(torch::stack(stateSeqSamples));
    //nextStateSeqBatch.push_back(torch::stack(nextStateSeqSamples));
    //actionSeqBatch.push_back(torch::stack(actionSeqSamples));
    //nextActionSeqBatch.push_back(torch::stack(nextActionSeqSamples));
  }

  auto rndEndIndicies = std::vector<size_t>();
  rndEndIndicies.reserve(rndStartIndices.size());
  std::transform(rndStartIndices.cbegin(), rndStartIndices.cend(), std::back_inserter(rndEndIndicies),
    [sequenceLen](size_t elem) { return elem+sequenceLen-1; }
  );
  auto rndEndIndicesPlus1 = std::vector<size_t>();
  rndEndIndicesPlus1.reserve(rndEndIndicies.size());
  std::transform(rndEndIndicies.cbegin(), rndEndIndicies.cend(), std::back_inserter(rndEndIndicesPlus1),
    [](size_t elem) { return elem+1; }
  );

  // These all need to be sampled AT THE END of the sequence - they verify the final action/reward/done after
  // the sequence of states has been executed/exploited/explored
  auto nextStateBatch = ReplayMemory::sampleFromVec(rndEndIndicesPlus1, this->states);
  auto actionSamples  = ReplayMemory::sampleFromVec(rndEndIndicies, this->actions);
  auto rewardSamples  = ReplayMemory::sampleFromVec(rndEndIndicies, this->rewards);
  auto doneSamples    = ReplayMemory::sampleFromVec(rndEndIndicies, this->dones);

  return {
    torch::stack(stateSeqBatch),
    torch::stack(nextStateBatch),
    //torch::stack(nextStateSeqBatch),
    //torch::stack(actionSeqBatch),
    //torch::stack(nextActionSeqBatch),
    torch::stack(actionSamples),
    torch::stack(rewardSamples),
    torch::stack(doneSamples)
  };
}

#endif // __REPLAYMEMORY_HPP__