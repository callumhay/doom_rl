#ifndef __RNG_HPP__
#define __RNG_HPP__

#include <cassert>
#include <random>
#include <memory>

class RNG;
using RNGPtr = std::shared_ptr<RNG>;

class RNG {
public:
  ~RNG() {};

  static RNGPtr getInstance() {
     if (instance == nullptr) {
      instance.reset(new RNG());
    }
    return instance;
  }

  auto getRngGen() { return this->rngGen; }

  // Generate a random unsigned integer number in [minIncl, maxIncl]
  size_t rand(size_t minIncl, size_t maxIncl) {
    std::uniform_int_distribution<size_t> rngDist(minIncl, maxIncl);
    return rngDist(this->rngGen);
  }

  // Generate a random real number in [minIncl, maxExcl)
  double rand(double minIncl, double maxExcl) {
    std::uniform_real_distribution<double> rngDist(minIncl, maxExcl);
    return rngDist(this->rngGen);
  }

  // Generate a random real number in [0,1)
  double randZeroToOne() { return this->rand(0.0, 1.0); }

  // Fisher–Yates_shuffle
  // size: Total number of indices to get back
  // maxSize: Total size of the potential pool of indices i.e., [0, maxSize) 
  // which will be placed inside the returned vector of indices
  std::vector<size_t> genShuffledIndices(size_t size, size_t maxSize) {
    assert(size <= maxSize);
    std::vector<size_t> res(size);

    for (auto i = 0; i != maxSize; ++i) {
      std::uniform_int_distribution<> dis(0, i);
      auto j = dis(this->rngGen);
      if (j < res.size()) {
        if (i < res.size()) {
          res[i] = res[j];
        }
        res[j] = i;
      }
    }
    return res;
  }


private:
  static RNGPtr instance;
  std::mt19937 rngGen;
  
  RNG() {
    std::random_device rd;
    this->rngGen = std::mt19937(rd());
  };
};

#endif // __RNG_HPP__