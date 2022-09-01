#ifndef __LEARNINGRATESCHEDULER_HPP__
#define __LEARNINGRATESCHEDULER_HPP__

#include <torch/torch.h>

class LearningRateScheduler : public torch::optim::LRScheduler {
public:
  LearningRateScheduler(torch::optim::Optimizer& optimizer): torch::optim::LRScheduler(optimizer) {};
  virtual ~LearningRateScheduler(){};

  virtual double getCurrentLR() const {
    return this->get_current_lrs()[0];
  };

  virtual void onBatchEnd(double loss) = 0;
  virtual void onEpochEnd(size_t epochNum) = 0;

};

#endif // __LEARNINGRATESCHEDULER_HPP__