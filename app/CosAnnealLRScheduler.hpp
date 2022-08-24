#ifndef __COSANNEALLRSCHEDULER_HPP__
#define __COSANNEALLRSCHEDULER_HPP__

#include <iostream>
#include <cmath>
#include <torch/torch.h>

constexpr double PI = 3.141592653589793238462643383279502884;

/**
 * Cosine annealing learning rate scheduler with periodic restarts.
 * Based on:
 * - https://www.jeremyjordan.me/nn-learning-rate/
 * - https://arxiv.org/abs/1608.03983
 */
class CosAnnealLRScheduler : public torch::optim::LRScheduler {
public:
  CosAnnealLRScheduler(
    torch::optim::Optimizer& optimizer, size_t expectedStepsPerEpoch, 
    double minLR=1e-5, double maxLR=1e-2, double lrDecay=0.999, 
    size_t cycleLength=10, double multFactor=1.5);

  ~CosAnnealLRScheduler() = default;

  size_t getAvgBatchesPerEpoch() { return static_cast<size_t>(this->avgBatchesPerEpoch); }
  double calcLR() const { return this->calcLR(this->batchSinceRestart); };

  // For the time being, we need to call these functions (see below) manually during training.
  // libtorch doesn't currently have much support for LR Scheduling.
  void onBatchEnd(double loss) {
    this->batchesInLastEpoch++;
    this->lossStepCount++;
    this->batchSinceRestart++;

    this->avgLoss += (loss-this->avgLoss) / static_cast<double>(this->lossStepCount);

    // Calculate the derivative (slope) of the loss once we have a previous value for it
    if (this->lossStepCount > 1) {
      auto dLoss = loss-this->prevLoss;
      this->dLossTotalInEpoch += dLoss;

      if (this->lossStepCount % 1000 == 0) { 
        std::cout << std::setprecision(6) << "Current Loss and Learning Rate: "
                  << "[step loss=" << loss << ", avg. loss=" << this->avgLoss << ", step dLoss=" << dLoss << ", epoch dLossTotal=" 
                  << this->dLossTotalInEpoch << ", lr=" << this->calcLR() << "]" << std::endl; 
      }
    }
    this->currLR = this->calcLR();
    this->prevLoss = loss;
  };
  void onEpochEnd(size_t epochNum) {

    // Make sure we update the cumulative average for the number of batches per epoch first,
    // this will be used if we increase/decrease the learning rate based on the total delta loss
    // in the epoch that just finished.
    this->avgBatchesPerEpoch += std::floor<int64_t>(static_cast<double>(
      this->batchesInLastEpoch-this->avgBatchesPerEpoch) / static_cast<double>(epochNum)
    );
    this->avgBatchesPerEpoch = std::max<int64_t>(1, this->avgBatchesPerEpoch);

    auto outputLrMsg = [this](const std::string& dirStr) {
      std::cout << "Auto adjusting learning rate schedule: " << dirStr << ". [Total delta loss for the last epoch was " 
                << std::fixed << std::setprecision(5) << this->dLossTotalInEpoch 
                << ", current avg. loss across all epochs is " << this->avgLoss << "]" << std::endl;
    };

    // Check for end of current cycle, apply restarts when necessary.
    if (epochNum+1 == this->nextRestart) {
      this->batchSinceRestart = 0;

      // If loss is decreasing or plateauing (on avg) then learning rate should be increasing
      if (this->dLossTotalInEpoch <= 0) {
        // Move to the other side of the cosine curve so that learning rate increases
        this->batchSinceRestart = std::floor<size_t>(this->batchesPerCycle() * this->fractToRestartIncreasing());
        outputLrMsg("Increasing");
      }

      this->cycleLength = std::ceil<size_t>(this->cycleLength * this->multFactor);
      this->nextRestart += this->cycleLength;
      this->maxLR *= this->lrDecay;
      this->lossStepCount = 0;
    }
    else {
      // Adjust the current batch number so that when we calculate the next learning rate
      // we will be moving in a suitable direction (i.e., the opposite direction of the loss):
      // Specifically:
      // - When loss is increasing significantly (on avg) then the learning rate should be decreasing
      // - otherwise, when loss is decreasing or plateauing significantly (on avg) then learning rate should be increasing
      if (this->dLossTotalInEpoch > 0) {
        auto bpc = this->batchesPerCycle();
        if (this->batchSinceRestart/bpc % 2 == 1) {
          // Move to the other side of the cosine curve so that learning rate decreases
          this->batchSinceRestart = std::floor<size_t>(bpc * this->fractToRestartDecreasing());
          outputLrMsg("Decreasing");
        }
      }
      // Place an extra restriction on prematurely increasing the learning rate, 
      // The loss overall should be low enough to justify the increase.
      else if (this->avgLoss < 0.1) { 
        auto bpc = this->batchesPerCycle();
        if (this->batchSinceRestart/bpc % 2 == 0) {
          // Move to the other side of the cosine curve so that learning rate increases
          this->batchSinceRestart = std::floor<size_t>(bpc * this->fractToRestartIncreasing());
          outputLrMsg("Increasing");
        }
      }
    }

    this->dLossTotalInEpoch  = 0.0;
    this->batchesInLastEpoch = 0;
  };

private:
  double currLR, minLR, maxLR, lrDecay, multFactor;
  size_t cycleLength;
  size_t batchSinceRestart, nextRestart;

  int64_t avgBatchesPerEpoch;
  int64_t batchesInLastEpoch;

  double avgLoss;
  double dLossTotalInEpoch;
  double prevLoss; // Store the previous loss so we can calculate the derivative
  size_t lossStepCount;

  std::vector<double> get_lrs() override;

  // Calculate the learning rate
  size_t batchesPerCycle() const { return this->avgBatchesPerEpoch * this->cycleLength; }
  
  double calcLR(size_t batchNum) const {
    auto fractToRestart = static_cast<double>(batchNum) / static_cast<double>(this->batchesPerCycle());
    auto lr = this->minLR + 0.5 * (this->maxLR - this->minLR) * (1.0 + std::cos(fractToRestart * PI));
    return lr;
  }

  double fractToRestartIncreasing() const {
    return (std::acos(1.0 - 2.0*(this->currLR-this->minLR)/(this->maxLR-this->minLR)) + PI) / PI;
  }
  double fractToRestartDecreasing() const {
    return std::acos(2.0*(this->currLR-this->minLR)/(this->maxLR-this->minLR) - 1.0) / PI;
  }

};

#endif // __COSANNEALLRSCHEDULER_HPP__
