#ifndef __CONVNETWORK_HPP__
#define __CONVNETWORK_HPP__

#include <torch/torch.h>

class ConvNetworkImpl : public torch::nn::Module {
public:
  ConvNetworkImpl(size_t inputChannels, size_t inputFrameHeight, size_t inputFrameWidth);

  size_t getOutputSize() const { return this->outputSize; }
  torch::Tensor forward(torch::Tensor input) { return this->convSeq(input); }

private:
  size_t outputSize;
  torch::nn::Sequential convSeq;
};

TORCH_MODULE(ConvNetwork);

#endif // __CONVNETWORK_HPP__