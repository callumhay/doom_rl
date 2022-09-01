#ifndef __TIMEDISTRIBUTED_HPP__
#define __TIMEDISTRIBUTED_HPP__

#include <torch/torch.h>

template<typename T>
class TimeDistributedImpl : public torch::nn::Module {
public:
  TimeDistributedImpl(T module, bool batchFirst): batchFirst(batchFirst) {
    this->module = this->register_module("inner_module", module);
  };

  torch::Tensor forward(torch::Tensor x) {
    auto xSizes = x.sizes();
    auto batchSize = xSizes[0];
    auto timeSteps = xSizes[1];
    auto c = xSizes[2];
    auto h = xSizes[3];
    auto w = xSizes[4];
    auto cIn  = x.view({batchSize*timeSteps, c, h, w});
    auto cOut = this->module->forward(cIn);
    auto rIn  = cOut.view({batchSize, timeSteps, -1});
    if (!batchFirst) { rIn = rIn.permute({1,0,2}); }
    return rIn;
  }

private:
  T module{nullptr};
  bool batchFirst;
};

template<typename T>
class TimeDistributed : public torch::nn::ModuleHolder<TimeDistributedImpl<T>> {
public:
  using torch::nn::ModuleHolder<TimeDistributedImpl<T>>::ModuleHolder;
  using Impl = TimeDistributedImpl<T>;
};


#endif // __TIMEDISTRIBUTED_HPP__
