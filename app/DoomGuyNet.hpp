#ifndef __DOOMGUYNET_HPP__
#define __DOOMGUYNET_HPP__

#include <memory>
#include <sstream>

#include <torch/torch.h>
#include <ATen/autocast_mode.h>

class DoomGuyNet : public torch::nn::Module {
public:
  static constexpr size_t version = 1;

  enum class Model {
    Online,
    Target
  };

  DoomGuyNet(size_t inputChannels, size_t outputDim, size_t version=DoomGuyNet::version);

  // Synchronize the target with the current online network
  void syncTarget() {
    std::stringstream stream;
    torch::save(this->online, stream);
    torch::load(this->target, stream); // Load the online network into the target network
    this->freezeTarget();
  };
  void freezeTarget() { 
    for (auto& p : this->target->parameters()) { p.set_requires_grad(false); }
  };

  torch::Tensor forward(torch::Tensor input, Model model) {
    // TODO: autocast is an unstable feature for half-precision, rewrite when a stable one exists!
    // See https://github.com/pytorch/pytorch/issues/44710 for updates.
    //at::autocast::set_enabled(true);
    auto result = model == Model::Online ? this->online->forward(input) : this->target->forward(input);
    //at::autocast::clear_cache();
    //at::autocast::set_enabled(false);
    return result;
  };

  //std::shared_ptr<Module> clone(const optional<Device>& device = nullopt) const override;
  void pretty_print(std::ostream& stream) const override;

private:
  size_t currVersion;
  size_t inputChannels, outputDim;
  torch::nn::Sequential online;
  torch::nn::Sequential target;
  
  void buildNetworks(size_t version);
  static torch::nn::Sequential buildV0Network(size_t inputChannels, size_t outputDim);
  static torch::nn::Sequential buildV1Network(size_t inputChannels, size_t outputDim);
};


#endif //__DOOMGUYNET_HPP__
