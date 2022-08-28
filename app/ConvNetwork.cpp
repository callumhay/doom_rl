#include "ConvNetwork.hpp"
#include "ConvUtils.hpp"

using Conv2dParams = std::array<int, 4>; // {input_channels, output_channels, kernel_size, stride}

auto conv2dGenerator(size_t inputWidth, size_t inputHeight, const Conv2DParams& params) {
  return std::make_tuple({
    torch::nn::Conv2d(torch::nn::Conv2dOptions(params[0], params[1], params[2]).stride(params[3])),
    ConvUtils::calcShapeConv2d(inputWidth, inputHeight, params[1], params[2], params[3])    
  });
}

ConvNetworkImpl::ConvNetworkImpl(size_t inputChannels, size_t inputFrameHeight, size_t inputFrameWidth) {
  // NOTE: Basic case for doom_rl is a framebuffer of [channels, height, width] = [3,200,320] 

  constexpr int numConv2ds = 6;
  constexpr std::array<Conv2dParams, numConv2ds> conv2dParams;
  conv2dParams[0] = {inputChannels,    256, 24, 1}; // 297,177
  conv2dParams[1] = {conv2dParams[0][1], 256, 24, 2}; // 137,77
  conv2dParams[2] = {conv2dParams[1][1], 384, 24, 3}; // 38,18
  conv2dParams[3] = {conv2dParams[2][1], 384, 12, 1}; // 27,7
  conv2dParams[4] = {conv2dParams[3][1], 384, 6,  1}; // 22,2
  conv2dParams[5] = {conv2dParams[4][1], 384, 2,  1}; // 21,1  -> 8064

  auto conv2dPipelineGenerator = [=]() {
    std::vector<torch::nn::Conv2d> conv2ds;
    conv2ds.reserve(numConv2ds);

    auto [conv2d, outputShape] = conv2dGenerator(inputFrameWidth, inputFrameHeight, conv2d0Params);
    conv2ds.push_back(conv2d);

    for (auto i = 1; i < numConv2ds; i++) {
      [conv2d, outputShape] = conv2dGenerator(outputShape[0], outputShape[1], conv2dParams[i]);
      conv2ds.push_back(conv2d);
    }

    return std::make_tuple(conv2ds, outputShape);
  };

  // Build the sequential module...
  auto [conv2ds, finalShape] = conv2dPipelineGenerator();
  for (auto& conv2d : conv2ds) {
    this->convSeq.push_back(conv2d);
    this->convSeq.push_back(torch::nn::ReLU());
  }
  this->convSeq.push_back(torch::nn::Flatten());

  this->outputSize = static_cast<size_t>(finalShape[0]*finalShape[1]*finalShape[2]);
}
