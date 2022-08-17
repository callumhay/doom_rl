#ifndef __RNG_HPP__
#define __RNG_HPP__

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

private:
  static RNGPtr instance;
  std::mt19937 rngGen;
  
  RNG() {
    std::random_device rd;
    this->rngGen = std::mt19937(rd());
  };
};

#endif // __RNG_HPP__