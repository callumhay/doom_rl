#ifndef __DOOMENVIRONMENT_HPP__
#define __DOOMENVIRONMENT_HPP__

#include <unordered_map>
#include <vector>
#include <mlpack/prereqs.hpp>

#include "ViZDoomTypes.h"

#include "DoomRewardVariable.hpp"

namespace vizdoom {
  class DoomGame;
}

class DoomEnvironment {
public:

  // A state is represented by the game's raw framebuffer RGB (24 bits) at 320x200 resolution
  class State {
  public:
    static constexpr size_t INPUT_WIDTH  = 320; // Width in pixels
    static constexpr size_t INPUT_HEIGHT = 200; // Height in pixels
    static constexpr size_t NUM_CHANNELS = 3;   // RGB as floats
    static constexpr size_t DIMENSION    = INPUT_WIDTH*INPUT_HEIGHT*NUM_CHANNELS;

    State(auto screenBuf) : screenBuf(screenBuf), data(DIMENSION, arma::fill::zeros) {}

    // For modifying the internal representation of the state.
    arma::colvec& Data() { return data; }

    // Encode the state
    const arma::colvec& Encode() const { return data; }

  private:
    vizdoom::ImageBufferPtr screenBuf;
    arma::colvec data; // TODO: Make this a cube(width,height,channels)???
  };

  class Action {
  public:
    enum actions {
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
    
    Action::actions action; // The currently selected action
    static constexpr size_t size = actions::DoomActionMoveForwardAndAttack + 1;
  };

  DoomEnvironment(size_t maxSteps);
  ~DoomEnvironment() { delete this->game; }

  double Sample(const State& state, const Action& action, State& nextState);
  double Sample(const State& state, const Action& action);
  State InitialSample(); // TODO: const State& return type???
  bool IsTerminal(const State& state) const;

  // Get the number of steps performed.
  size_t StepsPerformed() const { return this->stepsPerformed; }
  // Get the maximum number of steps allowed.
  size_t MaxSteps() const { return this->maxSteps; }
  // Set the maximum number of steps allowed.
  size_t& MaxSteps() { return this->maxSteps; }

private:
  size_t maxSteps;
  size_t stepsPerformed;

  vizdoom::DoomGame* game;
  std::unordered_map<Action::actions, std::vector<double>> actionMap;
  std::unordered_map<vizdoom::GameVariable, std::shared_ptr<DoomRewardVariable>> rewardVars;

  void initGameOptions();
  void initGameActions();
  void initGameVariables();
};

#endif // __DOOMENVIRONMENT_HPP__