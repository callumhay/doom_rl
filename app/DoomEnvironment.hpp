#ifndef __DOOMENVIRONMENT_HPP__
#define __DOOMENVIRONMENT_HPP__

#include <assert.h>
#include <unordered_map>
#include <vector>
#include <memory>
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
    // NOTE: The width and height must match one of the settings for ViZDoomGame! (see ViZDoomGame.cpp)
    static constexpr size_t INPUT_WIDTH  = 320; // Width in pixels
    static constexpr size_t INPUT_HEIGHT = 200; // Height in pixels
    static constexpr size_t NUM_CHANNELS = 3;   // RGB
    static constexpr size_t dimension    = INPUT_WIDTH*INPUT_HEIGHT*NUM_CHANNELS;

    State() : screenBuf(nullptr), data(dimension, arma::fill::zeros) {};
    explicit State(const arma::colvec& data): screenBuf(nullptr), data(data) {};
    explicit State(vizdoom::ImageBufferPtr screenBuf) : screenBuf(screenBuf), data(dimension, arma::fill::zeros) {
      if (screenBuf != nullptr) { this->updateDataFromBuffer(); }
    };

    State(const State&& rhs): screenBuf(std::move(rhs.screenBuf)), data(std::move(rhs.data)) {};
    State& operator=(State&& rhs) {
      if (this != &rhs) {
        this->screenBuf = std::move(rhs.screenBuf);
        this->data = std::move(rhs.data);
      }
      return *this;
    };

    // We need to define copying for mlpack
    State(const State& rhs) : screenBuf(rhs.screenBuf), data(rhs.data) {};
    State& operator=(const State& rhs) {
      if (this != &rhs) {
        this->screenBuf = rhs.screenBuf;
        this->data = rhs.data;
      }
      return *this;
    };

    // For modifying the internal representation of the state.
    void SetScreenBuf(vizdoom::ImageBufferPtr buf) {
      this->screenBuf = buf;
      this->updateDataFromBuffer();
    }

    arma::colvec& Data() { return data; }
    // Encode the state as a column vector (this MUST be encoded as a armadillo colvec!)
    const arma::colvec& Encode() { return data; }

  private:
    vizdoom::ImageBufferPtr screenBuf;
    arma::colvec data;

    void updateDataFromBuffer() {
      assert(this->screenBuf != nullptr);
      // Update the encoded data from the current screenBuf - this is a translation of
      // a std::shared_ptr<std::vector<uint8_t>> to an Armadillo Col<double>

      // NOTE: Images are interpreted as a stored vector of (width x height x channels).
      this->data = arma::colvec(std::vector<double>(screenBuf->begin(), screenBuf->end()));
    }
  };

  class Action {
  public:
    enum Actions {
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
    
    Action::Actions action; // The currently selected action
    static constexpr size_t size = static_cast<size_t>(Actions::DoomActionMoveForwardAndAttack) + 1;
  };

  DoomEnvironment(vizdoom::DoomGame* game, size_t maxSteps);

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

  void reset();

private:
  size_t maxSteps;
  size_t stepsPerformed;
  size_t frameSkip;

  vizdoom::DoomGame* game;
  std::unordered_map<Action::Actions, std::vector<double>> actionMap;
  std::unordered_map<vizdoom::GameVariable, std::shared_ptr<DoomRewardVariable>> rewardVars;

  void initGameOptions();
  void initGameActions();
  void initGameVariables();
};

#endif // __DOOMENVIRONMENT_HPP__