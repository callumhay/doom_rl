#ifndef __DEBUG_DOOM_RL_HPP__
#define __DEBUG_DOOM_RL_HPP__

#ifdef NDEBUG
  template <typename T>
  inline void printTensor(const T& t);
#else
#define DEBUG
  #include <torch/torch.h>
  #include <iostream>
  inline void printTensor(const torch::Tensor& t) { std::cout << t << std::endl; }
#endif


#endif // __DEBUG_DOOM_RL_HPP__
