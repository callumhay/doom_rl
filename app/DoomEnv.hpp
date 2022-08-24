#ifndef __DOOMENV_HPP__
#define __DOOMENV_HPP__

#include <assert.h>
#include <iostream>
#include <memory>
#include <tuple>
#include <array>
#include <unordered_map>

#include <torch/torch.h>

#include "ViZDoom.h"
#include "ViZDoomTypes.h"
#include "DoomRewardVariable.hpp"
#include "DoomGuyNet.hpp"

class DoomEnv {
public:

  class State {
  public:
    static constexpr int NUM_CHANNELS = 3;   // RGB

    static constexpr int SCREEN_BUFFER_WIDTH  = 320; // Original screen buffer width in pixels
    static constexpr int SCREEN_BUFFER_HEIGHT = 200; // Original screen buffer height in pixels

    static constexpr std::array<int,2> TENSOR_INPUT_HEIGHT_WIDTH() {
      switch (DoomGuyNet::version) {
        default:
        case 0: return {100, 160};
        case 1: return {SCREEN_BUFFER_HEIGHT, SCREEN_BUFFER_WIDTH};
      }
    }

    State(vizdoom::ImageBufferPtr screenBuf);

    torch::Tensor& tensor() { return this->screenTensor; }

  private:
    // Framebuffer image tensor that represents the state
    torch::Tensor screenTensor;
  };

  enum class Action {
    // Base actions
    DoomActionMoveLeft = 0,
    DoomActionMoveRight,
    DoomActionTurnLeft,
    DoomActionTurnRight,
    DoomActionAttack,
    DoomActionMoveBackward,
    DoomActionMoveForward,
    DoomActionUse,
    // Combo actions
    DoomActionMoveLeftAndAttack,
    DoomActionMoveRightAndAttack,
    DoomActionMoveBackwardAndAttack,
    DoomActionMoveForwardAndAttack,
  };
  static constexpr size_t numActions = static_cast<size_t>(Action::DoomActionMoveForwardAndAttack) + 1;

  DoomEnv(size_t maxSteps=1e4, size_t frameSkip=4, const std::string& mapName="E1M1");
  ~DoomEnv();

  using StatePtr = std::shared_ptr<DoomEnv::State>;
  using StepInfo = std::tuple<StatePtr, double, bool>;
  StatePtr reset();
  StepInfo step(const Action& a);

  size_t getStepsPerformed() const { return this->stepsPerformed; }
  size_t getMaxSteps() const { return this->maxSteps; }

  void setMap(const std::string& mapName) { this->doomMapToLoadNext = mapName; };
  void setCycledMap();
  void setRandomMap();

  const std::string& getMapToLoadNext() const { return this->doomMapToLoadNext; }

private:
  std::unique_ptr<vizdoom::DoomGame> game;
  std::unordered_map<Action, std::vector<double>> actionMap;
  std::vector<std::shared_ptr<DoomRewardVariable>> rewardVars;
  StatePtr lastState;
  size_t stepsPerformed;

  // Configuration Variables
  size_t frameSkip;
  size_t maxSteps;
  std::string doomMapToLoadNext;

  bool isEpisodeFinished() const;

  void initGameOptions();
  void initGameActions();
  void initGameVariables();
};

inline DoomEnv::~DoomEnv() { this->game->close(); }

// Start a new episode of play/training
inline DoomEnv::StatePtr DoomEnv::reset() {
  this->stepsPerformed = 0;

  //this->game->close();
  this->game->setDoomMap(this->doomMapToLoadNext);
  this->game->newEpisode();
  //this->game->init();

  // (Re)initialize all our variables (used to track rewards)
  for (auto& varPtr : this->rewardVars) { varPtr->reinit(*this->game); }

  // Grab the very first state of the game and return it
  auto gameState = this->game->getState();
  assert(gameState != nullptr);
  this->lastState = std::make_shared<DoomEnv::State>(gameState->screenBuffer);
  return this->lastState;
}

/**
 * Execute a single action and step forward to the next state.
 * @returns An std::tuple of the form <nextState:State, reward:double, epsiodeFinished:bool>
 */
inline DoomEnv::StepInfo DoomEnv::step(const DoomEnv::Action& a) {
  this->stepsPerformed++;
  auto reward = 0.0;
  auto gameActionVec = this->actionMap[a];

  // Calculate the current action and total state reward based on our reward variables
  // NOTE: You can make a "prolonged" action and skip frames by providing the 2nd arg to makeAction
  try {
    reward += this->game->makeAction(gameActionVec, this->frameSkip); // This will advance the ViZDoom game state
    if (reward != 0.0) {
      std::cout << "Reward granted from ViZDoom gym 'makeAction()' [Reward: " << reward << "]" << std::endl;
    }
  }
  catch (...) {
    // If the game window was closed we'll get an exception here, exit gracefully
    std::cout << "Game window was closed. Terminating program." << std::endl;
    exit(0);
  }

  // TODO?
  // std::vector<ImageBufferPtr> stateBuffers;
  // stateBuffers.reserve(this->frameSkip);
  // for (auto i = 0; i < this->frameSkip; i++) { 
  //  reward += this->game->makeAction(gameActionVec, 1);
  //  auto gameState = this->game->getState();
  //  assert(gameState != nullptr);
  //  stateBuffers.push_back(gameState->screenBuffer);
  // }
  auto done = this->isEpisodeFinished();
  auto gameState = this->game->getState();
  if (gameState != nullptr) {
    for (auto& varPtr : this->rewardVars) { 
      reward += varPtr->updateAndCalculateReward(*this->game);
    }
    this->lastState = std::make_shared<DoomEnv::State>(gameState->screenBuffer);
  }
  
  // If the map end is reached the agent gets a big fat reward
  if (this->game->isMapEnded()) { // NOTE: isMapEnded was added to the DoomGame class manually and ViZDoom was recompiled with it!
    reward += 1000.0;
  }

  return std::make_tuple(this->lastState, reward, done);
}


#endif // __DOOMENV_HPP__