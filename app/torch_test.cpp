
#include <iostream>
#include <cstdlib>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "DoomEnv.hpp"
#include "DoomGuy.hpp"
#include "DoomRLLogger.hpp"

int main() {
  std::srand(42);
  auto env = std::make_unique<DoomEnv>();




  /*
  std::vector<uint8_t> fb = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
  auto opts = torch::TensorOptions().dtype(torch::kUInt8);
  auto fbTensor = torch::from_blob(fb.data(), {4,4}, opts).clone();
  std::cout << fbTensor << std::endl;
  // Grab the center 2x2 of fbTensor
  using namespace torch::indexing;
  auto fbCenter2x2 = fbTensor.index({Slice(1,3), Slice(1,3)});
  std::cout << fbCenter2x2 << std::endl;

  std::cout << fbTensor.permute({1,0}).unsqueeze(0) << std::endl;

  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  auto oneValTensor = torch::tensor({1});
  auto item = oneValTensor.item();
  std::cout << typeid(item).name() << std::endl;
  */
}