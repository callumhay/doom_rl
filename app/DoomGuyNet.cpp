#include "DoomGuyNet.hpp"

DoomGuyNet::DoomGuyNet(size_t inputChannels, size_t outputDim, size_t version) : 
currVersion(version), inputChannels(inputChannels), outputDim(outputDim) {
  this->buildNetworks(this->currVersion);
}

/*
std::shared_ptr<Module> DoomGuyNet::clone(const optional<Device>& device = nullopt) const {
  auto cloned = std::make_shared<DoomGuyNet>(this->inputChannels, this->outputDim, this->currVersion);
  {
    std::stringstream stream;
    torch::save(this->online, stream);
    torch::load(cloned->online, stream);
  }
  {
    std::stringstream stream;
    torch::save(this->target, stream);
    torch::load(cloned->target, stream);
  }

  if (device != nullopt) { cloned->to(device); }
  return cloned;
}
*/

void DoomGuyNet::pretty_print(std::ostream& stream) const {
  stream << "Target and Online Network Architecture: " << std::endl;
  stream << this->online << std::endl;
}

void DoomGuyNet::buildNetworks(size_t version) {
  assert(version <= DoomGuyNet::version); // Check for unsupported version

  torch::nn::Sequential onlineNet;
  torch::nn::Sequential targetNet;
  std::cout << "Building online and target networks (DoomRL agent network v" << version << ")" << std::endl;
  switch (DoomGuyNet::version) {
    case 0:
      onlineNet = DoomGuyNet::buildV0Network(this->inputChannels, this->outputDim);
      targetNet = DoomGuyNet::buildV0Network(this->inputChannels, this->outputDim);
      break;
    
    case 1:
      onlineNet = DoomGuyNet::buildV1Network(this->inputChannels, this->outputDim);
      targetNet = DoomGuyNet::buildV1Network(this->inputChannels, this->outputDim);
      break;

    default:
      assert(false);
      return;
  }

  this->target = this->register_module("target", onlineNet);
  this->online = this->register_module("online", targetNet);

  // Q-Target parameters are frozen
  for (auto& p : this->target->parameters()) { p.set_requires_grad(false); }
  this->currVersion = version;

  this->pretty_print(std::cout);
}

torch::nn::Sequential DoomGuyNet::buildV0Network(size_t inputChannels, size_t outputDim) {
  constexpr int layer0OutChannels = 32;
  constexpr int layer1OutChannels = 64;
  constexpr int layer2OutChannels = 64;

  return torch::nn::Sequential({
    {"v0_conv2d0_3to32_8x8_4", torch::nn::Conv2d(torch::nn::Conv2dOptions(inputChannels, layer0OutChannels, 8).stride(4))},
    {"v0_leakyrelu0", torch::nn::LeakyReLU()},
    {"v0_conv2d1_32to64_5x5_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(layer0OutChannels, layer1OutChannels, 5).stride(2))},
    {"v0_leakyrelu1", torch::nn::LeakyReLU()},
    {"v0_conv2d2_64to64_3x3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(layer1OutChannels, layer2OutChannels, 3).stride(1))},
    {"v0_leakyrelu2", torch::nn::LeakyReLU()},
    {"v0_flatten0", torch::nn::Flatten()},
    {"v0_linear0_8192_1024", torch::nn::Linear(torch::nn::LinearOptions(8192, 1024))},
    {"v0_relu0", torch::nn::ReLU()},
    {"v0_linear1_1024_out", torch::nn::Linear(torch::nn::LinearOptions(1024, outputDim))}
  });
}

torch::nn::Sequential DoomGuyNet::buildV1Network(size_t inputChannels, size_t outputDim) {
  constexpr int conv0OutChannels = 3;
  constexpr int conv1OutChannels = 96;
  constexpr int conv2OutChannels = 256;
  constexpr int conv3OutChannels = 384;
  constexpr int conv4OutChannels = 256;

  return torch::nn::Sequential({
    {"v1_conv2d0", torch::nn::Conv2d(torch::nn::Conv2dOptions(inputChannels, conv0OutChannels, 7).stride(2))}, // [w,h,c] = [320,200,3] -> [157,97,3]
    {"v1_relu0", torch::nn::ReLU()},
    {"v1_maxpool0", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({4,4}).stride({2,2}))}, // [77,47,3]
    {"v1_conv2d1", torch::nn::Conv2d(torch::nn::Conv2dOptions(conv0OutChannels, conv1OutChannels, 5).stride(2))}, // [37,22,96]
    {"v1_relu1", torch::nn::ReLU()},
    {"v1_maxpool1", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({3,3}).stride({2,2}))}, // [18,10,256]
    {"v1_conv2d2", torch::nn::Conv2d(torch::nn::Conv2dOptions(conv1OutChannels, conv2OutChannels, 3).stride(1))}, // [16,8,256]
    {"v1_relu2", torch::nn::ReLU()},
    {"v1_conv2d3", torch::nn::Conv2d(torch::nn::Conv2dOptions(conv2OutChannels, conv3OutChannels, 3).stride(1))}, // [14,6,384]
    {"v1_relu3", torch::nn::ReLU()},
    {"v1_conv2d4", torch::nn::Conv2d(torch::nn::Conv2dOptions(conv3OutChannels, conv4OutChannels, 3).stride(1))}, // [12,4,256]
    {"v1_relu4", torch::nn::ReLU()},
    {"v1_maxpool2", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({2,2}).stride({1,1}))}, // [11,3,256]
    {"v1_flatten0", torch::nn::Flatten()}, // [8448]
    {"v1_linear0_7680_4096",torch::nn::Linear(torch::nn::LinearOptions(8448, 4096))},
    {"v1_relu5", torch::nn::ReLU()},
    {"v1_linear1_4096_4096", torch::nn::Linear(torch::nn::LinearOptions(4096, 4096))},
    {"v1_relu6", torch::nn::ReLU()},
    {"v1_linear2_4096_out", torch::nn::Linear(torch::nn::LinearOptions(4096, outputDim))}
  });
}