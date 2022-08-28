#ifndef __DOOMGUYNET_HPP__
#define __DOOMGUYNET_HPP__

#include <memory>
#include <sstream>

#include <torch/torch.h>
#include <ATen/autocast_mode.h>

class DoomGuyNetImpl : public torch::nn::Module {
public:
  static constexpr size_t version = 2;

  enum class Model {
    Online,
    Target
  };

  DoomGuyNetImpl(size_t inputChannels, size_t outputDim, size_t version=DoomGuyNetImpl::version);

  size_t getCurrVersion() const { return this->currVersion; }
  size_t getOutputDim() const { return this->outputDim; }

  // Synchronize the target with the current online network
  void syncTarget() {
    std::stringstream stream;
    torch::save(this->online, stream);
    torch::load(this->target, stream); // Load the online network into the target network
    this->freezeTarget();
  };
  void initNetworkParams() { 
    //should be unnecessary: for (auto& p : this->online->parameters()) { p.set_requires_grad(true); }
    this->freezeTarget();
  };
  void freezeTarget() {
    this->target->eval(); // The target is never training
    for (auto& p : this->target->parameters()) { p.set_requires_grad(false); }
  }

  torch::Tensor forward(torch::Tensor input, Model model) {
    return model == Model::Online ? this->online->forward(input) : this->target->forward(input);
  };

  void train(bool on = true) override {
    this->online->train(on); // The online network is the only thing ever trained
  }

  //std::shared_ptr<Module> clone(const optional<Device>& device = nullopt) const override;
  void pretty_print(std::ostream& stream) const override;

  torch::nn::Sequential online{nullptr};
  torch::nn::Sequential target{nullptr};

private:
  size_t currVersion;
  size_t inputChannels, outputDim;

  void buildNetworks(size_t version);
  static torch::nn::Sequential buildV0Network(size_t inputChannels, size_t outputDim);
  static torch::nn::Sequential buildV1Network(size_t inputChannels, size_t outputDim);
  static torch::nn::Sequential buildV2Network(size_t inputChannels, size_t outputDim);
  static torch::nn::Sequential buildV3Network(size_t inputChannels, size_t outputDim);
};

TORCH_MODULE(DoomGuyNet);

#endif //__DOOMGUYNET_HPP__
