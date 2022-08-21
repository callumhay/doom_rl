#ifndef __DOOMGUYNET_HPP__
#define __DOOMGUYNET_HPP__

#include <memory>
#include <sstream>

#include <torch/torch.h>
#include <ATen/autocast_mode.h>

class DoomGuyNet : public torch::nn::Module {
public:
  enum class Model {
    Online,
    Target
  };

  DoomGuyNet(torch::Tensor inputDim, size_t outputDim);

  // Synchronize the target with the current online network
  void syncTarget() {
    std::stringstream stream;
    torch::save(this->online, stream);
    torch::load(this->target, stream); // Load the online network into the target network
    // Q-Target parameters are frozen
    for (auto& p : this->target->parameters()) { p.set_requires_grad(false); }
  };

  torch::Tensor forward(torch::Tensor input, Model model) {
    // TODO: autocast is an unstable feature for half-precision, rewrite when a stable one exists!
    // See https://github.com/pytorch/pytorch/issues/44710 for updates.
    at::autocast::set_enabled(true);
    auto result = model == Model::Online ? this->online->forward(input) : this->target->forward(input);
    at::autocast::clear_cache();
    at::autocast::set_enabled(false);
    return result;
  };

private:
  torch::nn::Sequential online;
  torch::nn::Sequential target;
  
  torch::nn::Sequential buildNetwork(size_t inputChannels, size_t outputDim);

};


#endif //__DOOMGUYNET_HPP__
