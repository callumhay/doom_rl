#include "modules/ResidualNetwork.hpp"
#include "modules/TimeDistributed.hpp"
#include "modules/QNetworkLSTM.hpp"
#include "modules/ConvNetwork.hpp"

#include "DoomEnv.hpp"
#include "DoomGuyNet.hpp"


DoomGuyNetImpl::DoomGuyNetImpl(size_t inputChannels, size_t outputDim, size_t version, size_t minorVersion) : 
currVersion(version), currMinorVersion(minorVersion), inputChannels(inputChannels), 
outputDim(outputDim), online(nullptr), target(nullptr) {
  this->buildNetworks(this->currVersion, this->currMinorVersion);
}

// 'Shoehorn' the network in the given 'otherNet' into this network
void DoomGuyNetImpl::shoehorn(const DoomGuyNet& otherNet) {
  torch::NoGradGuard no_grad;

  auto otherParams = otherNet->online->named_parameters(true);
  auto params = this->online->named_parameters(true);
  auto buffers = this->online->named_buffers(true);

  for (auto& val : otherParams) {
    auto name = val.key();
    auto* t = params.find(name);
    if (t != nullptr) {
      t->copy_(val.value());
    }
    else {
      t = buffers.find(name);
      if (t != nullptr) {
        t->copy_(val.value());
      }
    }
  }
  
  this->syncTarget();
}

void DoomGuyNetImpl::pretty_print(std::ostream& stream) const {
  stream << "Target and Online Network Architecture: " << std::endl;
  stream << this->online << std::endl;
}

void DoomGuyNetImpl::buildNetworks(size_t version, size_t minorVersion) {
  assert(version <= DoomGuyNetImpl::version); // Check for unsupported version

  torch::nn::Sequential onlineNet;
  torch::nn::Sequential targetNet;
  std::cout << "Building online and target networks (DoomRL agent network v" << version << ")" << std::endl;
  std::function<torch::nn::Sequential(size_t, size_t, size_t)> buildFunc;
  switch (DoomGuyNetImpl::version) {
    case 0: buildFunc = DoomGuyNetImpl::buildV0Network; break;
    case 1: buildFunc = DoomGuyNetImpl::buildV1Network; break;
    case 2: buildFunc = DoomGuyNetImpl::buildV2Network; break;
    case 3: buildFunc = DoomGuyNetImpl::buildV3Network; break;
    case 4: buildFunc = DoomGuyNetImpl::buildV4Network; break;

    default:
      assert(false);
      return;
  }

  this->online = this->register_module("online", buildFunc(this->inputChannels, this->outputDim, minorVersion));
  this->target = this->register_module("target", buildFunc(this->inputChannels, this->outputDim, minorVersion));

  this->initNetworkParams();
  this->currVersion = version;
  this->currMinorVersion = minorVersion;
}

torch::nn::Sequential DoomGuyNetImpl::buildV0Network(size_t inputChannels, size_t outputDim, size_t minorVersion) {
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

torch::nn::Sequential DoomGuyNetImpl::buildV1Network(size_t inputChannels, size_t outputDim, size_t minorVersion) {
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

torch::nn::Sequential DoomGuyNetImpl::buildV2Network(size_t inputChannels, size_t outputDim, size_t minorVersion) {
 return torch::nn::Sequential(
    torch::nn::Conv2d(torch::nn::Conv2dOptions(inputChannels, 256, 8).stride(4)),
    torch::nn::LeakyReLU(),
    torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 4).stride(3)),
    torch::nn::LeakyReLU(),
    torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(2)),
    torch::nn::LeakyReLU(),
    torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 128, 2).stride(1)),
    torch::nn::LeakyReLU(),
    torch::nn::Flatten(),
    torch::nn::Dropout(0),
    torch::nn::Linear(torch::nn::LinearOptions(8448, 4096)),
    torch::nn::LeakyReLU(),
    torch::nn::Dropout(0.2),
    torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)),
    torch::nn::ReLU(),
    torch::nn::Dropout(0.2),
    torch::nn::Linear(torch::nn::LinearOptions(4096, outputDim))
 );
}

torch::nn::Sequential DoomGuyNetImpl::buildV3Network(size_t inputChannels, size_t outputDim, size_t minorVersion) {
  auto [w,h] = DoomEnv::State::getNetInputSize(3);
  assert(w == h);

  constexpr auto resNetOutSize  = 1024;

  return torch::nn::Sequential(
    // NOTE: ResNet50 uses layers {3,4,6,3}
    // We're currently using ResNet-18 for speed
    ResidualNetwork(ResidualNetworkImpl::Layers({2,2,2,2}), resNetOutSize, inputChannels, 64, 1, w),
    torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.02).inplace(true)),
    torch::nn::Dropout(0.0),
    torch::nn::Linear(torch::nn::LinearOptions(resNetOutSize, outputDim))
  );
}

torch::nn::Sequential DoomGuyNetImpl::buildV4Network(size_t inputChannels, size_t outputDim, size_t minorVersion) {
  auto [w,h] = DoomEnv::State::getNetInputSize(4);
  assert(w == h);

  //constexpr auto resNetOutSize  = 1024;
  constexpr auto lstmHidden = 128;

  // NOTE: The number of parameters in a single LSTM layer is 4*(h*(h+i)+h), 
  // where i is inputs and h is hidden size... this gets big fast!
  auto convNet = ConvNetwork(inputChannels, h, w);
  return torch::nn::Sequential(
    TimeDistributed<ConvNetwork>(convNet, true),
    QNetworkLSTM(convNet->getOutputSize(), outputDim, lstmHidden, 2, 0.2)
  );
}
