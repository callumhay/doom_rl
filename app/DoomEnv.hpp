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

    static std::array<int,2> TENSOR_INPUT_HEIGHT_WIDTH(size_t networkVersion) {
      switch (networkVersion) {
        case 0: case 3: return {100, 160};
        case 1: case 2: return {SCREEN_BUFFER_HEIGHT, SCREEN_BUFFER_WIDTH};
        default: assert(false); break;
      }
      return {0,0};
    }

    static torch::Tensor buildStateTensor(vizdoom::ImageBufferPtr screenBuf, size_t networkVersion);
    static torch::Tensor buildEmptyStateTensor(size_t networkVersion);

  private:
    State(){}
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

    //DoomNoAction
  };
  static size_t getNumActions(size_t doomGuyNetVersion) {
    return static_cast<size_t>(doomGuyNetVersion < 2 ? Action::DoomActionMoveForwardAndAttack : Action::DoomActionUse) + 1;
  }

  DoomEnv(size_t maxSteps=1e4, size_t frameSkip=4, bool activePlayEnabled=false, const std::string& mapName="E1M1");
  ~DoomEnv();

  using StepInfo = std::tuple<torch::Tensor, double, bool>;
  torch::Tensor reset(size_t networkVersion);
  StepInfo step(const Action& a, size_t networkVersion);

  size_t getStepsPerformed() const { return this->stepsPerformed; }
  size_t getMaxSteps() const { return this->maxSteps; }

  void setMap(const std::string& mapName) { this->doomMapToLoadNext = mapName; };
  void setCycledMap();
  void setRandomMap();

  const std::string& getMapToLoadNext() const { return this->doomMapToLoadNext; };

  bool isInActivePlayMode() const { return this->game->getMode() == vizdoom::Mode::SPECTATOR; };
  Action getLastAction() const;
  void advanceActionFrames() { this->game->advanceAction(this->frameSkip); }

private:
  std::unique_ptr<vizdoom::DoomGame> game;
  std::unordered_map<Action, std::vector<double>> actionMap;
  std::vector<std::shared_ptr<DoomRewardVariable>> rewardVars;
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
inline torch::Tensor DoomEnv::reset(size_t networkVersion) {
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

  return DoomEnv::State::buildStateTensor(gameState->screenBuffer, networkVersion);
}

#endif // __DOOMENV_HPP__