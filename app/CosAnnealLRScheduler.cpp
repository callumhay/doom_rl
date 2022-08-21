#include <algorithm>
#include "CosAnnealLRScheduler.hpp"

using namespace torch::optim;

/*
  minLR: The lower bound of the learning rate range for the experiment.
  maxLR: The upper bound of the learning rate range for the experiment.
  stepsPerEpoch: Number of mini-batches in the dataset. Calculated as 
                 `std::ceil(static_cast<double>(epoch_size)/static_cast<double>(batch_size))`. 
  lrDecay: Reduce the max_lr after the completion of each cycle.
           Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
  cycleLength: Initial number of epochs in a cycle.
  multFactor: Scale epochs_to_restart after each full cycle completion.
*/
CosAnnealLRScheduler::CosAnnealLRScheduler(
  Optimizer& optimizer, size_t stepsPerEpoch, 
  double minLR, double maxLR, double lrDecay, 
  size_t cycleLength, double multFactor
):
LRScheduler(optimizer), stepsPerEpoch(stepsPerEpoch), minLR(minLR), maxLR(maxLR),
lrDecay(lrDecay), cycleLength(cycleLength), multFactor(multFactor),
batchSinceRestart(0), nextRestart(cycleLength), lossStepCount(0), avgdLossValue(0) {
  auto lrs = get_current_lrs();
  this->currLR = lrs[0];

  // Calculate where we are in the cycle based on the initial LR, this is based on the calcLR() method,
  // but rearranged to calculate the batchSinceRestart so that our learning rate is increasing to start
  auto ftrIncreasing = this->fractToRestartIncreasing();
  this->batchSinceRestart = std::floor<size_t>(this->batchesPerCycle() * ftrIncreasing);
}

std::vector<double> CosAnnealLRScheduler::get_lrs() {
  auto lrs = this->get_current_lrs();
  std::transform(lrs.begin(), lrs.end(), lrs.begin(), [this](const double& v) {
    return this->currLR;
  });
  return lrs;
}
