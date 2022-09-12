#ifndef __REPLAYMEMORY_HPP__
#define __REPLAYMEMORY_HPP__

#include <cassert>
#include <algorithm>
#include <vector>
#include <array>


#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/version.hpp>

#include <torch/torch.h>

#include "utils/TensorUtils.hpp"

// https://acsweb.ucsd.edu/~wfedus/pdf/replay.pdf
class ReplayMemory {
public:
  static constexpr size_t DEFAULT_REPLAY_BATCH_SIZE = 32;

  ReplayMemory(size_t capacity): currStateIdx(0), currNonStateIdx(0) { this->initCapacity(capacity); }
  ReplayMemory(size_t stateWidth, size_t stateHeight, size_t stateChannels): 
    ReplayMemory(std::max<size_t>(
      REPLAY_MEMORY_MIN_SIZE, (REPLAY_MEMORY_FP_RATIO_FPS/(stateWidth*stateHeight*stateChannels+6)) * REPLAY_MEMORY_FP_RATIO_SIZE
    )
  ){};

  ReplayMemory(const ReplayMemory&) = delete;
  ReplayMemory& operator=(const ReplayMemory&) = delete;
  ~ReplayMemory() = default;

  size_t getCacheSize() { return this->dones.size(); }
  void initState(torch::Tensor initState);
  void cache(torch::Tensor nextState, int action, double reward, bool done);

  using ReplayData = std::array<torch::Tensor, 5>;
  using SequenceReplayData = std::array<torch::Tensor, 5>;

  // Produces an array of 5 tensors, each tensor with a shape of [batchSize, ...feature_values]
  // The 5 tensors correspond to, in order: [indices, stateBatch, nextStateBatch, actionBatch, rewardBatch, doneBatch]
  std::pair<std::vector<size_t>, ReplayData> randomRecall(size_t batchSize) { return this->randomSamples(batchSize); }

  // Produces a pair, the first part of the pair is the indices choosen, and the second is 
  // an array of 5 specialized tensors: [stateSeq, nextStateBatch, actionBatch, rewardBatch, doneBatch]
  // The first tensor is the indices within this ReplayMemory that were choosen.
  // The second 2 tensors are formatted such that they capture a sequence of length 'sequenceLen'.
  // The last 3 tensors are formatted similar to the randomRecall function, capturing the rewards and dones
  // for the state corresponding to the end of the sequence (last state) of the stateSeq tensor.
  std::pair<std::vector<size_t>, SequenceReplayData> sequenceRecall(size_t batchSize, size_t sequenceLen);

  // After a training cycle through a batch of recalled samples/sequences, call this to update the batches' indices
  // with prioritized recall based on the calculated TD difference
  void updateIndexTDDiffs(const std::vector<size_t>& indices, torch::Tensor absTDDiffs);

private:
  using TensorVec = std::vector<torch::Tensor>;
  
  // If the capacity of the replay memory is too big we WILL run out of memory !!
  static constexpr double REPLAY_MEMORY_FP_RATIO_FPS  = (320*200*3 + 6);
  static constexpr double REPLAY_MEMORY_FP_RATIO_SIZE = 35000;
  static constexpr size_t REPLAY_MEMORY_MIN_SIZE      = 70000;

  // Key where not-yet assigned TD Difference values are stored in the priority map
  static constexpr int UNASSIGNED_TD_DIFF = -1;

  // Randomness with which we grab unassigned indices
  static constexpr double UNASSIGNED_PROB = 0.5;

  size_t capacity;
  size_t currStateIdx, currNonStateIdx;

  TensorVec states;
  TensorVec actions;
  TensorVec rewards;
  TensorVec dones;
  std::vector<double> tdDiffs; // |TD_difference| for each index so we can quickly look-up in the priority map

  // Map |TD_difference| values for indices that have been choosen so that we can prioritize
  // the replay of high error values and learn faster
  std::map<double, std::set<size_t>, std::greater<double>> priorityMap;

  void initCapacity(size_t capacity) {
    assert(capacity > 0);
    this->capacity = capacity;
    std::cout << "Replay memory capacity set to " << this->capacity << std::endl;

    this->states.reserve(this->capacity);
    this->actions.reserve(this->capacity);
    this->rewards.reserve(this->capacity);
    this->dones.reserve(this->capacity);
    this->tdDiffs.reserve(this->capacity);

    // Initialize a set for the unassigned key (since it will be used anytime cache is called)
    this->priorityMap[UNASSIGNED_TD_DIFF] = std::set<size_t>();
  }

  // Get a vector of random indicies where each element in the vector is an array of size 2, the first element
  // is the random index and the next is that same index + 1. The choice of indices is limited by the given
  // sequenceLen (i.e., the choices will be made far enough back in the history of the replay memory to support
  // the given sequenceLen). 
  auto getPriorityRandomIndices(size_t batchSize, size_t sequenceLen=1) {
    assert(
      this->states.size() >= batchSize+1 && this->actions.size() >= batchSize+1 && 
      this->rewards.size() >= batchSize+1 && this->dones.size() >= batchSize+1
    );
    assert(sequenceLen > 0 && batchSize > 0);
    assert(sequenceLen+batchSize-1 <= this->dones.size());

    auto maxUnassigned = this->priorityMap[UNASSIGNED_TD_DIFF].size();
    auto maxAssigned   = this->dones.size()-maxUnassigned;

    // Grab some random unassigned indicies based on some probability
    auto numUnassigned = std::min<size_t>(maxUnassigned, std::max<size_t>(1, static_cast<size_t>(batchSize*UNASSIGNED_PROB)));
    // Make sure we get enough indices to satisfy the batch size (in case there aren't enough assigned values)
    auto numAssigned = batchSize - numUnassigned;
    if (maxAssigned < numAssigned) {
      numAssigned   = maxAssigned;
      numUnassigned = batchSize - numAssigned; 
    }

    auto rndIndices = std::vector<size_t>();
    rndIndices.reserve(batchSize);

    auto elementIdxIsValid = [this, sequenceLen](auto elemIdx) {
      // Make sure we aren't trying to get a sequence that overlaps the current insertion index
      // in the case where we've reached capacity (i.e., have started overwriting the buffers)
      auto reachedCapacity = this->dones.size() == this->capacity;
      auto wrapAroundEndIdx = (elemIdx+sequenceLen-1) % this->capacity;
      auto reachedCapacityAndNoIdxOverlaps = reachedCapacity &&
        !((elemIdx < this->currNonStateIdx || elemIdx > wrapAroundEndIdx) && wrapAroundEndIdx >= this->currNonStateIdx);
      // If we haven't reached capacity make sure that we aren't sequencing from past the end of the buffers
      // Since we need to grab the next state after the sequence, it will be the limiting consideration
      auto notReachedCapacityAndNoOverflow = !reachedCapacity && elemIdx+sequenceLen < this->states.size();

      return reachedCapacityAndNoIdxOverlaps || notReachedCapacityAndNoOverflow;
    };
    auto elementIdxIsContiguous = [this, sequenceLen](auto elemIdx) {
      // Special case to test for: If the sequence contains a 'done' (i.e., an index of the done buffer is true)
      // within the sequence, then we don't use it
      for (auto i = 0; i < sequenceLen; i++) {
        auto idx = (elemIdx+i) % this->capacity;
        auto doneVal = this->dones[idx][0].item().toInt();
        if (doneVal == 1) { return false; }
      }
      return true;
    };

    const auto unassignedMapIter = this->priorityMap.find(UNASSIGNED_TD_DIFF);
    const auto& unassignedSet = unassignedMapIter->second;
    for (auto iter = unassignedSet.cbegin(); numUnassigned > 0 && iter != unassignedSet.cend(); iter++) {
      auto elemIdx = *iter;
      // Make sure we take an index that can support the sequenceLen without breaking the sequence or going out of bounds
      if (elementIdxIsValid(elemIdx) && elementIdxIsContiguous(elemIdx)) {
        rndIndices.push_back(elemIdx);
        numUnassigned--;
      }
    }

    numAssigned += numUnassigned;

    for (auto mapIter = this->priorityMap.cbegin(); numAssigned > 0 && mapIter != std::prev(this->priorityMap.cend()); mapIter++) {
      const auto& set = mapIter->second;
      for (auto setIter = set.cbegin(); numAssigned > 0 && setIter != set.cend(); setIter++) {
        auto elemIdx = *setIter;
        // Make sure we take an index that can support the sequenceLen without breaking the sequence or going out of bounds
        if (elementIdxIsValid(elemIdx) && elementIdxIsContiguous(elemIdx)) {
          rndIndices.push_back(elemIdx);
          numAssigned--;
        }
      }
    }

    assert(rndIndices.size() == batchSize);

    return rndIndices;
  }

  // Produces an array of [state, nextState, actions, rewards, dones], where each element is a torch::Tensor
  // of the specified batch size
  std::pair<std::vector<size_t>, ReplayData> randomSamples(size_t batchSize) {

    auto rndIndices = this->getPriorityRandomIndices(batchSize);
    auto rndIndicesPlus1 = std::vector<size_t>();
    rndIndicesPlus1.reserve(rndIndices.size());
    std::transform(rndIndices.cbegin(), rndIndices.cend(), std::back_inserter(rndIndicesPlus1), 
      [this](size_t elem) { return (elem+1) % this->capacity; }
    );

    auto stateSamples     = ReplayMemory::sampleFromBuf(rndIndices, this->states);
    auto nextStateSamples = ReplayMemory::sampleFromBuf(rndIndicesPlus1, this->states);
    auto actionSamples    = ReplayMemory::sampleFromBuf(rndIndices, this->actions);
    auto rewardSamples    = ReplayMemory::sampleFromBuf(rndIndices, this->rewards);
    auto doneSamples      = ReplayMemory::sampleFromBuf(rndIndices, this->dones);

    return std::make_pair(
      rndIndices, ReplayData({
      torch::stack(stateSamples),
      torch::stack(nextStateSamples),
      torch::stack(actionSamples),
      torch::stack(rewardSamples),
      torch::stack(doneSamples)
    }));
  };

  static TensorVec sampleFromBuf(const std::vector<size_t>& randomIndices, TensorVec& sampleBuf) {
    TensorVec samples;
    samples.reserve(randomIndices.size());
    for (auto idx: randomIndices) {
      samples.push_back(sampleBuf[idx]);
    }
    return samples;
  };

  TensorVec sequenceFromBuf(size_t startIdx, size_t sequenceLen, TensorVec& sampleBuf) {
    TensorVec seqSamples;
    seqSamples.reserve(sequenceLen);
    for (auto i = startIdx; i < (startIdx+sequenceLen); i++) {
      seqSamples.push_back(sampleBuf[i%this->capacity]);
    }
    assert(seqSamples.size() == sequenceLen);
    return seqSamples;
  };

  // Serialization **************************************************
  friend class boost::serialization::access;
  template<class Archive>
  void save(Archive & ar, const unsigned int version) const {
    ar & this->capacity;
    ar & this->currStateIdx;
    ar & this->currNonStateIdx;
    ar & this->priorityMap;
    ar & this->states;
    ar & this->actions;
    ar & this->rewards;
    ar & this->dones;
    ar & this->tdDiffs;
  }

  template<class Archive>
  void load(Archive & ar, const unsigned int version) {
    ar & this->capacity;

    this->states = TensorVec();
    this->actions = TensorVec();
    this->rewards = TensorVec();
    this->dones = TensorVec();
    this->initCapacity(this->capacity);
    
    ar & this->currStateIdx;
    ar & this->currNonStateIdx;
    ar & this->priorityMap;
    ar & this->states;
    ar & this->actions;
    ar & this->rewards;
    ar & this->dones;
    ar & this->tdDiffs;
  }
  BOOST_SERIALIZATION_SPLIT_MEMBER()

}; // class ReplayMemory

inline void ReplayMemory::initState(torch::Tensor initState) {
  if (this->states.size() == this->capacity) { this->states[this->currStateIdx] = initState; }
  else { this->states.push_back(initState); }
  this->currStateIdx = (this->currStateIdx+1) % this->capacity;
}

inline void ReplayMemory::cache(torch::Tensor nextState, int action, double reward, bool done) {
  // Handle the case where we're overwriting previous indices, we'll need to
  // track down the previous index in our priorityMap and remove it
  if (this->dones.size() == this->capacity) {
    auto iter = this->priorityMap.find(this->tdDiffs[this->currNonStateIdx]);
    assert(iter != this->priorityMap.cend());
    iter->second.erase(this->currNonStateIdx);

    this->actions[this->currNonStateIdx] = torch::tensor({action});
    this->rewards[this->currNonStateIdx] = torch::tensor({reward});
    this->dones[this->currNonStateIdx]   = torch::tensor({done ? 1.0 : 0.0});
    this->tdDiffs[this->currNonStateIdx] = UNASSIGNED_TD_DIFF;
  }
  else {
    this->actions.push_back(torch::tensor({action}));
    this->rewards.push_back(torch::tensor({reward}));
    this->dones.push_back(torch::tensor({done ? 1.0 : 0.0}));
    this->tdDiffs.push_back(UNASSIGNED_TD_DIFF);
  }
  this->priorityMap[UNASSIGNED_TD_DIFF].insert(this->currNonStateIdx);

  if (!done) {
    if (this->states.size() == this->capacity) { this->states[this->currStateIdx] = nextState; }
    else { this->states.push_back(nextState); }
    this->currStateIdx = (this->currStateIdx+1) % this->capacity;
  }
  this->currNonStateIdx = (this->currNonStateIdx+1) % this->capacity;
};

inline std::pair<std::vector<size_t>, ReplayMemory::SequenceReplayData> 
ReplayMemory::sequenceRecall(size_t batchSize, size_t sequenceLen) {

  assert(
    this->states.size() >= batchSize && this->actions.size() >= batchSize && 
    this->rewards.size() >= batchSize && this->dones.size() >= batchSize
  );

  auto rndStartIndices = this->getPriorityRandomIndices(batchSize, sequenceLen);
  TensorVec stateSeqBatch;
  stateSeqBatch.reserve(batchSize);

  // Build the sequences and as we go add them to the batches
  for (auto& rndIdx : rndStartIndices) {
    // Generate a sequence vector
    auto stateSeqSamples = this->sequenceFromBuf(rndIdx, sequenceLen, this->states);
    // We need to satisfy the following tensor shape for each batch of sequences:
    // [batch_size, sequence_length, features]
    stateSeqBatch.push_back(torch::stack(stateSeqSamples));
  }

  auto rndEndIndicies = std::vector<size_t>();
  rndEndIndicies.reserve(rndStartIndices.size());
  std::transform(rndStartIndices.cbegin(), rndStartIndices.cend(), std::back_inserter(rndEndIndicies),
    [sequenceLen, this](size_t elem) { return (elem+sequenceLen-1) % this->capacity; }
  );
  auto rndEndIndicesPlus1 = std::vector<size_t>();
  rndEndIndicesPlus1.reserve(rndEndIndicies.size());
  std::transform(rndEndIndicies.cbegin(), rndEndIndicies.cend(), std::back_inserter(rndEndIndicesPlus1),
    [this](size_t elem) { return (elem+1) % this->capacity; }
  );

  // These all need to be sampled AT THE END of the sequence - they verify the final action/reward/done after
  // the sequence of states has been executed/exploited/explored
  auto nextStateBatch = ReplayMemory::sampleFromBuf(rndEndIndicesPlus1, this->states);
  auto actionSamples  = ReplayMemory::sampleFromBuf(rndEndIndicies, this->actions);
  auto rewardSamples  = ReplayMemory::sampleFromBuf(rndEndIndicies, this->rewards);
  auto doneSamples    = ReplayMemory::sampleFromBuf(rndEndIndicies, this->dones);

  return std::make_pair(
    rndStartIndices, SequenceReplayData({
    torch::stack(stateSeqBatch),
    torch::stack(nextStateBatch),
    torch::stack(actionSamples),
    torch::stack(rewardSamples),
    torch::stack(doneSamples)
  }));
}

inline void ReplayMemory::updateIndexTDDiffs(const std::vector<size_t>& indices, torch::Tensor absTDDiffs) {
  auto cpuAbsTDDiffs = absTDDiffs.cpu();
  assert(indices.size() == cpuAbsTDDiffs.sizes()[0] && cpuAbsTDDiffs.sizes().size() == 2 && cpuAbsTDDiffs.sizes()[1] == 1);

  for (auto i = 0; i < indices.size(); i++) {
    const auto idx = indices[i]; 
    const auto tdDiff = cpuAbsTDDiffs[i][0].item<double>();
    const auto prevTDDiff = this->tdDiffs[idx];
    assert(tdDiff >= 0);

    if (tdDiff == prevTDDiff) { continue; }

    // Remove the previous value from the priority map
    auto prevIter = this->priorityMap.find(prevTDDiff);
    if (prevIter != this->priorityMap.cend()) {
      prevIter->second.erase(idx);
      if (prevIter->second.size() == 0) {
        this->priorityMap.erase(prevIter);
      }
    }

    // Update the priority map and TD Difference values with the new one
    auto findIter = this->priorityMap.find(tdDiff);
    if (findIter == this->priorityMap.cend()) {
      this->priorityMap[tdDiff] = std::set<size_t>();
    }
    this->priorityMap[tdDiff].insert(idx);
    this->tdDiffs[idx] = tdDiff;
  }
}

#endif // __REPLAYMEMORY_HPP__