#include "DoomGuyNet.hpp"

DoomGuyNet::DoomGuyNet(torch::Tensor inputDim, size_t outputDim) {
  auto inChannels = static_cast<size_t>(inputDim[0].item<int>());
  this->target = this->register_module("target", this->buildNetwork(inChannels, outputDim));
  this->online = this->register_module("online", this->buildNetwork(inChannels, outputDim));

  // Q-Target parameters are frozen
  for (auto& p : this->target->parameters()) { p.set_requires_grad(false); }
}

torch::nn::Sequential DoomGuyNet::buildNetwork(size_t inputChannels, size_t outputDim) {
  constexpr int layer0OutChannels = 32;
  constexpr int layer1OutChannels = 64;
  constexpr int layer2OutChannels = 64;
  auto model = torch::nn::Sequential(



    torch::nn::Conv2d(torch::nn::Conv2dOptions(inputChannels, layer0OutChannels, 8).stride(4)), // (inChannels, outChannels, kernelSize)
    torch::nn::LeakyReLU(), // [w,h,c] = [39,24,32]
    torch::nn::Conv2d(torch::nn::Conv2dOptions(layer0OutChannels, layer1OutChannels, 5).stride(2)),
    torch::nn::LeakyReLU(), // [18,10,64]
    
    //torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}).stride({2,2})), // [9,5,64]
    //torch::nn::Conv2d(torch::nn::Conv2dOptions(layer1OutChannels, layer2OutChannels, 2).stride(1)), // [8,4,64]

    torch::nn::Conv2d(torch::nn::Conv2dOptions(layer1OutChannels, layer2OutChannels, 3).stride(1)),
    torch::nn::LeakyReLU(), // [16,8,64]
    torch::nn::Flatten(), // -> [8192]
    torch::nn::Linear(torch::nn::LinearOptions(8192, 1024)),
    torch::nn::ReLU(),
    torch::nn::Linear(torch::nn::LinearOptions(1024, outputDim))
  );

  // Cast all the layers in the model to half-precision
  //model->to(torch::kFloat16);

  return model;
}