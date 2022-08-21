#ifndef __COSANNEALLRSCHEDULER_HPP__
#define __COSANNEALLRSCHEDULER_HPP__

#include <iostream>
#include <numbers>
#include <torch/torch.h>

/**
 * Cosine annealing learning rate scheduler with periodic restarts.
 * Based on:
 * - https://www.jeremyjordan.me/nn-learning-rate/
 * - https://arxiv.org/abs/1608.03983
 */
class CosAnnealLRScheduler : public torch::optim::LRScheduler {
public:
  CosAnnealLRScheduler(
    torch::optim::Optimizer& optimizer, size_t stepsPerEpoch, 
    double minLR=1e-5, double maxLR=1e-2, double lrDecay=0.999, 
    size_t cycleLength=10, double multFactor=1.5);

  ~CosAnnealLRScheduler() = default;

  // For the time being, we need to call these functions (see below) manually during training.
  // libtorch doesn't currently have much support for LR Scheduling.
  void onBatchEnd(double loss) {
    this->lossStepCount++;
    this->batchSinceRestart++;

    // Calculate the derivative (slope) of the loss once we have a previous value for it
    if (this->lossStepCount > 1) {
      auto dLoss = loss-this->prevLoss;
      // Update the average delta loss (cumulative average)
      this->avgdLossValue += (dLoss-this->avgdLossValue) / static_cast<double>(this->lossStepCount-1);

      // Adjust the current batch number so that when we calculate the next learning rate
      // we will be moving in a suitable direction (i.e., the opposite direction of the loss):
      // Specifically:
      // - When loss is increasing (on avg) then the learning rate should be decreasing
      // - otherwise, when loss is decreasing or plateauing (on avg) then learning rate should be increasing
      if (this->avgdLossValue > 0) {
        auto bpc = this->batchesPerCycle();
        if (this->batchSinceRestart > bpc) {
          // Move to the other side of the cosine curve so that learning rate decreases
          this->batchSinceRestart = std::floor<size_t>(bpc * this->fractToRestartDecreasing());
          //std::cout << "Auto adjusting learning rate schedule: Decreasing." << std::endl;
        }
      }
      else {
        auto bpc = this->batchesPerCycle();
        if (this->batchSinceRestart < bpc) {
          // Move to the other side of the cosine curve so that learning rate increases
          this->batchSinceRestart = std::floor<size_t>(bpc * this->fractToRestartIncreasing());
          //std::cout << "Auto adjusting learning rate schedule: Increasing." << std::endl;
        }
      }

      if (this->lossStepCount % 100 == 0) { 
        std::cout << std::setprecision(6) << "Current Loss and Learning Rate: "
                  << "[loss=" << loss << ", dLoss=" << dLoss << ", dLossAvg=" << this->avgdLossValue << ", lr=" << this->calcLR() << "]" << std::endl; 
      }
    }
    this->currLR = this->calcLR();
    this->prevLoss = loss;
  };
  void onEpochEnd(size_t epochNum) {
    // Check for end of current cycle, apply restarts when necessary.
    if (epochNum+1 == this->nextRestart) {
      this->batchSinceRestart = 0;
      this->cycleLength = std::ceil<size_t>(this->cycleLength * this->multFactor);
      this->nextRestart += this->cycleLength;
      this->maxLR *= this->lrDecay;
      this->avgdLossValue = 0;
      this->lossStepCount = 0;
    }
  };

private:
  double currLR, minLR, maxLR, lrDecay, multFactor;
  size_t stepsPerEpoch, cycleLength;
  size_t batchSinceRestart, nextRestart;

  double prevLoss; // Store the previous loss so we can calculate the derivative
  double avgdLossValue;
  size_t lossStepCount;

  std::vector<double> get_lrs() override;

  // Calculate the learning rate
  size_t batchesPerCycle() const { return this->stepsPerEpoch * this->cycleLength; }
  double calcLR() const { return this->calcLR(this->batchSinceRestart); };
  double calcLR(size_t batchNum) const {
    auto fractToRestart = static_cast<double>(batchNum) / static_cast<double>(this->batchesPerCycle());
    auto lr = this->minLR + 0.5 * (this->maxLR - this->minLR) * (1.0 + std::cos(fractToRestart * std::numbers::pi));
    return lr;
  }

  double fractToRestartIncreasing() const {
    return (std::acos(1.0 - 2.0*(this->currLR-this->minLR)/(this->maxLR-this->minLR)) + std::numbers::pi) / std::numbers::pi;
  }
  double fractToRestartDecreasing() const {
    return std::acos(2.0*(this->currLR-this->minLR)/(this->maxLR-this->minLR) - 1.0) / std::numbers::pi;
  }

};

#endif // __COSANNEALLRSCHEDULER_HPP__
