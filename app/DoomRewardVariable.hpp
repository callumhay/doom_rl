#ifndef __DOOMREWARDVARIABLE_HPP__
#define __DOOMREWARDVARIABLE_HPP__

#include "ViZDoomTypes.h"
#include "ViZDoomGame.h"
#include <string>
#include <memory>

class DoomRewardVariable {
public: 
  virtual ~DoomRewardVariable(){}

  virtual void reinit(vizdoom::DoomGame* game) = 0;
  virtual double updateAndCalculateReward(vizdoom::DoomGame* game) = 0;

protected:
  DoomRewardVariable(){}
};

template <typename T, auto rewardFunc>
class DoomRewardVariableT : public DoomRewardVariable {
public:
  DoomRewardVariableT(vizdoom::GameVariable varType, std::string&& varDescription): 
    varType(varType), varDescription(varDescription), currValue(0) {}
  ~DoomRewardVariableT(){};

  void reinit(vizdoom::DoomGame* game) override {
    this->currValue = static_cast<T>(game->getGameVariable(this->varType));
    std::cout << "Initial " << this->varDescription << ":\t\t\t" << this->currValue << std::endl;
  }

  double updateAndCalculateReward(vizdoom::DoomGame* game) override {
    auto newValue = static_cast<T>(game->getGameVariable(this->varType));
    double reward = 0.0;
    if (newValue != this->currValue) {
      std::cout << this->varDescription << " changed to " << newValue << std::endl;
      reward = static_cast<double>(this->getReward(this->currValue, newValue));
      this->currValue = newValue;
    }
    return reward;
  }

private:
  vizdoom::GameVariable varType;
  std::string varDescription;
  T currValue;
  decltype(rewardFunc) getReward = rewardFunc;
};

template <auto rewardFunc>
using DoomRewardVarInt = DoomRewardVariableT<int, rewardFunc>;

//template <auto rewardFunc>
//using DoomRewardVarDbl = DoomRewardVariableT<double, rewardFunc>;

#endif // __DOOMREWARDVARIABLE_HPP__