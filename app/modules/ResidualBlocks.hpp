#ifndef __RESIDUALBLOCKS_HPP__
#define __RESIDUALBLOCKS_HPP__

#include <cassert>
#include <torch/torch.h>

namespace residual_block_util {

inline auto conv1x1(size_t inPlanes, size_t outPlanes, size_t stride=1) {
  return torch::nn::Conv2d(
    torch::nn::Conv2dOptions(inPlanes, outPlanes, 1).stride(stride).bias(false)
  );
};
inline auto conv3x3(size_t inPlanes, size_t outPlanes, size_t stride=1, size_t groups=1, size_t dilation=1) {
  return torch::nn::Conv2d(
    torch::nn::Conv2dOptions(inPlanes, outPlanes, 3).stride(stride).bias(false).groups(groups).padding(dilation).dilation(dilation)
  );
};
inline auto normLayer(size_t planes) {
  return torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes));
};

};

class ResidualBottleneckImpl : public torch::nn::Module {
public:
  inline static const size_t expansion = 4;

  ResidualBottleneckImpl(
    size_t inPlanes, size_t planes, size_t stride=1, torch::nn::Sequential downsample=nullptr,
    size_t groups=1, size_t baseWidth=64, size_t dilation=1
  ) : downsample(downsample) {

    using namespace residual_block_util;
    auto width = static_cast<size_t>(planes * (static_cast<double>(baseWidth)/64.0)) * groups;

    this->conv1 = this->register_module("conv1", conv1x1(inPlanes, width));
    this->bn1   = this->register_module("bn1", normLayer(width));
    this->conv2 = this->register_module("conv2", conv3x3(width, width, stride, groups, dilation));
    this->bn2   = this->register_module("bn2", normLayer(width));
    this->conv3 = this->register_module("conv3", conv1x1(width, planes*this->expansion));
    this->bn3   = this->register_module("bn3", normLayer(planes * this->expansion));
    this->relu  = this->register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));
    
    if (this->downsample) {
      this->downsample = this->register_module("downsample", this->downsample);
    }
  };

  torch::Tensor forward(torch::Tensor x) {
    auto residual = x;

    auto out = this->relu(this->bn1(this->conv1(x)));
    out = this->relu(this->bn2(this->conv2(out)));
    out = this->bn3(this->conv3(out));

    if (this->downsample) {
      residual = this->downsample->forward(x);
    }
    out += residual;
    out = this->relu(out);

    return out;
  };

private:
  torch::nn::Conv2d conv1{nullptr};
  torch::nn::Conv2d conv2{nullptr};
  torch::nn::Conv2d conv3{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::BatchNorm2d bn2{nullptr};
  torch::nn::BatchNorm2d bn3{nullptr};
  torch::nn::ReLU relu{nullptr};

  torch::nn::Sequential downsample;
};

TORCH_MODULE(ResidualBottleneck);

#endif // __RESIDUALBLOCK_HPP__
