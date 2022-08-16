#ifndef __DOOMGUYNET_HPP__
#define __DOOMGUYNET_HPP__

#include <torch/torch.h>

class DoomGuyNet : public torch::nn::Module {
public:
  enum class Model {
    Online,
    Target
  };

  DoomGuyNet(torch::Tensor inputDim, torch::Tensor outputDim);


private:
  torch::nn::Sequential online;
  torch::nn::Sequential target;
  
};

#endif //__DOOMGUYNET_HPP__
