#ifndef __DOOMREWARDVARIABLE_HPP__
#define __DOOMREWARDVARIABLE_HPP__

#include "ViZDoomTypes.h"
#include "ViZDoomGame.h"
#include <string>
#include <memory>

class DoomRewardVariable {
public: 
  virtual ~DoomRewardVariable(){}

  virtual void reinit(vizdoom::DoomGame& game) = 0;
  virtual double updateAndCalculateReward(vizdoom::DoomGame& game) = 0;

protected:
  DoomRewardVariable(const std::string& varDescription) : varDescription(varDescription) {};
  std::string varDescription;
};

template <typename T, auto rewardFunc>
class DoomRewardVariableT : public DoomRewardVariable {
public:
  DoomRewardVariableT(vizdoom::GameVariable varType, const std::string& varDescription):
    DoomRewardVariable(varDescription), varType(varType), currValue(0) {}

  void reinit(vizdoom::DoomGame& game) override {
    this->currValue = static_cast<T>(game.getGameVariable(this->varType));
    std::cout << "Initial " << std::left << std::setw(20) << (this->varDescription+":")  
              << std::setw(5) << this->currValue << std::endl;
  }

  double updateAndCalculateReward(vizdoom::DoomGame& game) override {
    auto newValue = static_cast<T>(game.getGameVariable(this->varType));
    double reward = 0.0;
    if (newValue != this->currValue) {
      reward = static_cast<double>(this->getReward(this->currValue, newValue));
      this->currValue = newValue;

      // TODO: Use log filtering here
      //std::cout << this->varDescription << " changed to " << newValue;
      //if (reward != 0.0) {
      //  std::cout << " [Reward: " << reward << "]";
      //}
      //std::cout << std::endl;

    }
    return reward;
  }

private:
  vizdoom::GameVariable varType;
  T currValue;
  decltype(rewardFunc) getReward = rewardFunc;
};

template <auto rewardFunc>
using DoomRewardVarInt = DoomRewardVariableT<int, rewardFunc>;

/**
 * Reward for movement / exploration through the world based on the initial position
 * at the start of an episode.
 */
class DoomPosRewardVariable : public DoomRewardVariable {
public:
  static constexpr double rewardRadiusDiff = 100.0;

  DoomPosRewardVariable(vizdoom::GameVariable varPosX, vizdoom::GameVariable varPosY, vizdoom::GameVariable varPosZ):
    DoomRewardVariable("Player position"), varPosX(varPosX), varPosY(varPosY), varPosZ(varPosZ), currMaxRadius(0) {}

  void reinit(vizdoom::DoomGame& game) override {
    this->currMaxRadius = 0.0;
    this->initX = static_cast<double>(game.getGameVariable(this->varPosX));
    this->initY = static_cast<double>(game.getGameVariable(this->varPosY));
    this->initZ = static_cast<double>(game.getGameVariable(this->varPosZ));
    std::cout << "Initial " << std::left << std::setw(20) << (this->varDescription+":")  
              << "(" << this->initX << ", " << this->initY << ", " << this->initZ << ")"
              << std::endl;
  }

  double updateAndCalculateReward(vizdoom::DoomGame& game) override {
    auto reward = 0.0;
    auto currX = static_cast<double>(game.getGameVariable(this->varPosX));
    auto currY = static_cast<double>(game.getGameVariable(this->varPosY));
    auto currZ = static_cast<double>(game.getGameVariable(this->varPosZ));

    // Calculate the squared distance from the initial position
    auto dist  = std::sqrt(std::pow(currX-this->initX, 2) + std::pow(currY-this->initY, 2) + std::pow(currZ-this->initZ, 2));
    auto rDiff = dist - this->currMaxRadius;

    // If the distance travelled is larger than the current largest distance by the rewardRadiusDiff then give a reward
    if (rDiff >= rewardRadiusDiff) {
      reward += 0.01*dist;
      std::cout << std::fixed << std::setprecision(1)
                << this->varDescription << " changed - distance increased by " << rDiff 
                << " (total distance from episode start: " << dist << ")"
                << " [Reward: " << reward << "]" << std::endl;
      this->currMaxRadius = dist;
    }

    return reward;
  }

private:
  vizdoom::GameVariable varPosX, varPosY, varPosZ;
  double initX, initY, initZ;
  double currMaxRadius;
};


#endif // __DOOMREWARDVARIABLE_HPP__