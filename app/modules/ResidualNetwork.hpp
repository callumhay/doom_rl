#ifndef __RESIDUALNETWORK_HPP__
#define __RESIDUALNETWORK_HPP__

#include <torch/torch.h>

#include "ResidualBlocks.hpp"

// Based off ResNet v1.5, see https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
class ResidualNetworkImpl : public torch::nn::Module {
public:
  using Layers = std::array<size_t,4>;

  ResidualNetworkImpl(
    const Layers& layers, size_t outputSize, size_t inputChannels=3, size_t startPlanes=64, size_t groups=1, size_t widthPerGroup=64
  ): startPlanes(startPlanes), groups(groups), baseWidth(widthPerGroup) {

    using namespace residual_block_util;

    this->conv1 = this->register_module("conv1", torch::nn::Conv2d(
      torch::nn::Conv2dOptions(inputChannels, startPlanes, 7).stride(2).padding(3).bias(false)
    ));
    this->bn1  = this->register_module("bn1", normLayer(startPlanes));
    this->relu = this->register_module("relu", torch::nn::ReLU(torch::nn::ReLUOptions().inplace(true)));

    this->maxpool = this->register_module("maxpool", torch::nn::MaxPool2d(
      torch::nn::MaxPool2dOptions({3,3}).stride({2,2}).padding({1,1})
    ));
    
    constexpr auto l4OutChs = 512;
    this->layer1  = this->register_module("layer1", this->makeLayer(64, layers[0]));
    this->layer2  = this->register_module("layer2", this->makeLayer(128, layers[1], 2));
    this->layer3  = this->register_module("layer3", this->makeLayer(256, layers[2], 2));
    this->layer4  = this->register_module("layer4", this->makeLayer(l4OutChs, layers[3], 2));
    this->avgpool = this->register_module("avgpool", torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1,1})));

    this->fc = this->register_module("fc_out", torch::nn::Linear(l4OutChs*ResidualBottleneckImpl::expansion, outputSize));

    for (auto& m : this->modules(false)) {
      if (auto* conv2d = m->as<torch::nn::Conv2d>()) {
        torch::nn::init::kaiming_normal_(conv2d->weight, 0, torch::kFanOut, torch::kReLU);
      }
      else if (auto* bn2d = m->as<torch::nn::BatchNorm2d>()) {
        torch::nn::init::constant_(bn2d->weight, 1.0);
        torch::nn::init::constant_(bn2d->bias, 0.0);
      }
    }
  };

  torch::Tensor forward(torch::Tensor x) {
    x = this->maxpool(this->relu(this->bn1(this->conv1(x))));

    x = this->layer1->forward(x);
    x = this->layer2->forward(x);
    x = this->layer3->forward(x);
    x = this->layer4->forward(x);
    x = this->avgpool(x);

    x = this->flatten(x);
    x = this->fc(x);

    return x;
  };

private:
  size_t startPlanes;
  size_t groups;
  size_t baseWidth;

  torch::nn::Conv2d conv1{nullptr};
  torch::nn::BatchNorm2d bn1{nullptr};
  torch::nn::ReLU relu{nullptr};
  torch::nn::MaxPool2d maxpool{nullptr};
  torch::nn::AdaptiveAvgPool2d avgpool{nullptr};

  torch::nn::Sequential layer1{nullptr};
  torch::nn::Sequential layer2{nullptr};
  torch::nn::Sequential layer3{nullptr};
  torch::nn::Sequential layer4{nullptr};

  torch::nn::Linear fc{nullptr};
  torch::nn::Flatten flatten;

  torch::nn::Sequential makeLayer(size_t planes, size_t blocks, size_t stride=1) {
    using namespace residual_block_util;
    constexpr auto dilation = 1;

    torch::nn::Sequential downsample;
    const auto planeExpansion = planes*ResidualBottleneckImpl::expansion;
    if (stride != 1 || this->startPlanes != planeExpansion) {
      downsample = torch::nn::Sequential(
        conv1x1(this->startPlanes, planeExpansion, stride),
        normLayer(planeExpansion)
      );
    }
    
    torch::nn::Sequential result;
    result->push_back("sublayer1", ResidualBottleneck(this->startPlanes, planes, stride, downsample, this->groups, this->baseWidth, dilation));
    this->startPlanes = planeExpansion;
    for (auto i = 1; i < blocks; i++) {
      const auto layerNum = i+1;
      result->push_back("sublayer"+std::to_string(layerNum), 
        ResidualBottleneck(this->startPlanes, planes, 1, nullptr, this->groups, this->baseWidth, dilation)
      );
    }

    return result;
  };

};

TORCH_MODULE(ResidualNetwork);

#endif // __RESIDUALNETWORK_HPP__