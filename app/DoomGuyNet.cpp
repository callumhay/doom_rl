#include "DoomGuyNet.hpp"

DoomGuyNet::DoomGuyNet(torch::Tensor inputDim, size_t outputDim) {
  auto inChannels = static_cast<size_t>(inputDim[0].item<int>());
  this->target = this->register_module("target", this->buildNetwork(inChannels, outputDim));
  this->online = this->register_module("online", this->buildNetwork(inChannels, outputDim));

  // Q-Target parameters are frozen
  for (auto& p : this->target->parameters()) {
    p.set_requires_grad(false);
  }
}

torch::nn::Sequential DoomGuyNet::buildNetwork(size_t inputChannels, size_t outputDim) {
  constexpr int layer0OutChannels = 32;
  constexpr int layer1OutChannels = 64;
  constexpr int layer2OutChannels = 64;
  return torch::nn::Sequential(
    torch::nn::Conv2d(torch::nn::Conv2dOptions(inputChannels, layer0OutChannels, 8).stride(4)), // (inChannels, outChannels, kernelSize)
    torch::nn::LeakyReLU(),  // [w,h,c] = [39,24,32]
    torch::nn::Conv2d(torch::nn::Conv2dOptions(layer0OutChannels, layer1OutChannels, 4).stride(2)),
    torch::nn::LeakyReLU(),  // -> [12,7,64]
    torch::nn::Conv2d(torch::nn::Conv2dOptions(layer1OutChannels, layer2OutChannels, 3).stride(1)),
    torch::nn::LeakyReLU(), // -> [10,5,64]
    torch::nn::Flatten(), // -> [3200]
    torch::nn::Linear(3200, 512),
    torch::nn::ReLU(),
    torch::nn::Linear(512, outputDim)
  );
}