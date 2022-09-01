#ifndef __CONSTANTLRSCHEDULER_HPP__
#define __CONSTANTLRSCHEDULER_HPP__

#include "LearningRateScheduler.hpp"

class ConstantLRScheduler : public LearningRateScheduler {
public:
  ConstantLRScheduler(torch::optim::Optimizer& optimizer, double lr): LearningRateScheduler(optimizer), lr(lr) {};
  double getCurrentLR() const override { return this->lr; }

  void onBatchEnd(double loss) override {};
  void onEpochEnd(size_t epochNum) override {};

private:
  double lr;

  std::vector<double> get_lrs() override {
    auto lrs = this->get_current_lrs();
    std::transform(lrs.begin(), lrs.end(), lrs.begin(), [this](const double& v) {
      return this->lr;
    });
    return lrs;
  };

};

#endif // __CONSTANTLRSCHEDULER_HPP__