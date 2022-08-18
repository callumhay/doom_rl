#ifndef __DOOMGUYNET_HPP__
#define __DOOMGUYNET_HPP__

#include <memory>
#include <sstream>
#include <torch/torch.h>

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
  };

  torch::Tensor forward(torch::Tensor input, Model model) {
    return model == Model::Online ? this->online->forward(input) : this->target->forward(input);
  };

private:
  torch::nn::Sequential online;
  torch::nn::Sequential target;
  
  torch::nn::Sequential buildNetwork(size_t inputChannels, size_t outputDim);

};


#endif //__DOOMGUYNET_HPP__
