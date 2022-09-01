#ifndef __CONVNETWORK_HPP__
#define __CONVNETWORK_HPP__

#include <torch/torch.h>

#include "ConvUtils.hpp"

class ConvNetworkImpl : public torch::nn::Module {
public:
  ConvNetworkImpl(size_t inputChannels, size_t inputFrameHeight, size_t inputFrameWidth) {
    constexpr int numConv2ds = 6;
    std::array<Conv2dParams, numConv2ds> conv2dParams;
    conv2dParams[0] = {inputChannels,      64,  7, 2, 3}; // 64x64
    conv2dParams[1] = {conv2dParams[0][1], 128, 7, 2, 3}; // 32x32
    conv2dParams[2] = {conv2dParams[1][1], 256, 7, 2, 3}; // 16x16
    conv2dParams[3] = {conv2dParams[2][1], 512, 7, 2, 3}; // 8x8
    conv2dParams[4] = {conv2dParams[3][1], 512, 5, 1, 0}; // 4x4
    conv2dParams[5] = {conv2dParams[4][1], 512, 3, 1, 0}; // 2x2 -> 2048


    this->convSeq = this->register_module("conv_seq", this->convSeq);

    auto conv2dPipelineGenerator = [=]() {
      auto [conv2d, outputShape] = conv2dGenerator(inputFrameWidth, inputFrameHeight, conv2dParams[0]);

      this->convSeq->push_back(conv2d);
      this->convSeq->push_back(torch::nn::BatchNorm2d( outputShape[2]));
      this->convSeq->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

      for (auto i = 1; i < numConv2ds; i++) {
        auto [a, b] = conv2dGenerator(outputShape[0], outputShape[1], conv2dParams[i]);
        conv2d = a; outputShape = b;

        this->convSeq->push_back(conv2d);
        this->convSeq->push_back(torch::nn::BatchNorm2d(outputShape[2]));
        this->convSeq->push_back(torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
      }

      return outputShape;
    };

    // Build the sequential module...
    auto finalShape = conv2dPipelineGenerator();
    // Flatten to the final output layer and record the size
    this->convSeq->push_back(torch::nn::Flatten());
    this->outputSize = finalShape[0]*finalShape[1]*finalShape[2];

    for (auto& m : this->modules(false)) {
      if (auto* conv2d = m->as<torch::nn::Conv2d>()) {
        torch::nn::init::kaiming_normal_(conv2d->weight, 0, torch::kFanOut, torch::kReLU);
      }
      else if (auto* bn2d = m->as<torch::nn::BatchNorm2d>()) {
        torch::nn::init::constant_(bn2d->weight, 1.0);
        torch::nn::init::constant_(bn2d->bias, 0.0);
      }
    }
  }

  size_t getOutputSize() const { return this->outputSize; }

  torch::Tensor forward(torch::Tensor input) { 
    return this->convSeq->forward(input); 
  }

private:
  using Conv2dParams = std::array<size_t, 5>; // {input_channels, output_channels, kernel_size, stride, padding}
  static std::tuple<torch::nn::Conv2d, ConvUtils::Conv2dOutputShape> conv2dGenerator(size_t inputWidth, size_t inputHeight, const Conv2dParams& params) {
    return std::make_tuple(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(params[0], params[1], params[2]).stride(params[3]).padding(params[4]).bias(false)),
      ConvUtils::calcShapeConv2d(inputWidth, inputHeight, params[1], params[2], params[3], params[4])    
    );
  }

  size_t outputSize;
  torch::nn::Sequential convSeq;
};

TORCH_MODULE(ConvNetwork);

#endif // __CONVNETWORK_HPP__