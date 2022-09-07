#ifndef __TENSORUTILS_HPP__
#define __TENSORUTILS_HPP__

#include <exception>
#include <vector>
#include <ostream>
#include <sstream>

#include <boost/serialization/split_free.hpp>
#include <boost/serialization/string.hpp>

#include <torch/torch.h>

class TensorUtils {
public:

  // Functionally equivalent to torchvision's functional normalize
  static torch::Tensor normalize(
    torch::Tensor tensor, torch::ArrayRef<double> mean, 
    torch::ArrayRef<double> std, bool inPlace
  );

  static void saveTensor(const torch::Tensor& t, const std::string& saveFilepath) { 
    auto bytes = torch::pickle_save(t);
    std::ofstream fout(saveFilepath, std::ios::out | std::ios::binary);
    fout.write(bytes.data(), bytes.size());
    fout.close();
  };

private:
  TensorUtils() = default;
  ~TensorUtils() = default;
};

inline torch::Tensor TensorUtils::normalize(
  torch::Tensor tensor, torch::ArrayRef<double> mean, torch::ArrayRef<double> std, bool inPlace
) {

  auto result = inPlace ? tensor : tensor.clone();
  auto tensorSize = tensor.sizes().size();
  if (tensorSize < 3) {
    throw std::invalid_argument("Expected tensor to be a tensor image of size (..., C, H, W). Got tensor.sizes().size() == " + std::to_string(tensorSize));
  }

  auto dtype = tensor.dtype();
  auto opts = torch::TensorOptions().dtype(dtype).device(tensor.device());
  auto tMean = torch::tensor(mean).to(opts);
  auto tStd  = torch::tensor(std).to(opts);
  if (torch::count_nonzero(tStd).sizes()[0] == tStd.sizes()[0]) {
    throw std::invalid_argument("std evalued to zero after conversion to "+torch::toString(opts)+" leading to division by zero.");
  }
  if (tMean.dim() == 1) {
    tMean = tMean.view({-1,1,1});
  }
  if (tStd.dim() == 1) {
    tStd = tStd.view({-1,1,1});
  }

  return result.sub_(tMean).div_(tStd);  
}

BOOST_SERIALIZATION_SPLIT_FREE(at::Tensor)
namespace boost {
namespace serialization {

template<class Archive>
void save(Archive& ar, const at::Tensor& t, unsigned int version) {
  std::stringstream ss;
  torch::save(t, ss);
  ar & ss.str();
}
template<class Archive>
void load(Archive& ar, at::Tensor& t, unsigned int version) {
  std::stringstream ss;
  std::string s;
  ar & s;
  ss.str(s);
  torch::load(t, ss);
}

} // namespace serialization
} // namespace boost


#endif // __TENSORUTILS_HPP__
