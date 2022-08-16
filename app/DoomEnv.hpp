#ifndef __DOOMENV_HPP__
#define __DOOMENV_HPP__

#include <assert.h>
#include <memory>
#include <tuple>
#include <unordered_map>

#include <torch/torch.h>

#include "ViZDoom.h"
#include "ViZDoomTypes.h"
#include "DoomRewardVariable.hpp"

class DoomEnv {
public:

  class State {
  public:
    static constexpr size_t NUM_CHANNELS = 3;   // RGB

    static constexpr size_t SCREEN_BUFFER_WIDTH  = 320; // Original screen buffer width in pixels
    static constexpr size_t SCREEN_BUFFER_HEIGHT = 200; // Original screen buffer height in pixels

    static constexpr size_t TENSOR_INPUT_WIDTH  = 160; // 1/2 the original screen buffer size
    static constexpr size_t TENSOR_INPUT_HEIGHT = 100;

    State(vizdoom::ImageBufferPtr screenBuf) {
      assert(screenBuf != nullptr);
      // Preprocess the game's framebuffer (stored as a flat array of uint8_t)...

      // Convert to a 3D tensor (height x width x channels)
      this->screenTensor = torch::from_blob(
        screenBuf->data(), 
        {SCREEN_BUFFER_HEIGHT, SCREEN_BUFFER_WIDTH, NUM_CHANNELS}, 
        torch::TensorOptions().dtype(torch::kUInt8)
      );

      // Reformat the tensor to be in the form that the interpolate function expects:
      // Batch x Channel x Height x Width
      this->screenTensor = this->screenTensor.permute({2,0,1}).unsqueeze(0); // shape is now [1,channels,height,width]

      // Downsample the screen buffer tensor - this will make a new tensor (so we don't need to clone anything)
      this->screenTensor = torch::nn::functional::interpolate(
        this->screenTensor, 
        torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>({TENSOR_INPUT_HEIGHT, TENSOR_INPUT_WIDTH}))
        .mode(torch::kBilinear)
      ); // shape is now [1, NUM_CHANNELS, TENSOR_INPUT_HEIGHT, TENSOR_INPUT_WIDTH]
      
      // Make sure to clone the data (we don't know what's going to happen to the original ImageBufferPtr after we leave)
      //this->screenTensor = this->screenTensor.clone();
    };

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

  DoomEnv(size_t frameSkip=4);
  ~DoomEnv();

  void reset();

  using StatePtr = std::unique_ptr<DoomEnv::State>;
  using StepInfo = std::tuple<StatePtr, double, bool>;
  StepInfo step(const Action& a);

private:
  std::unique_ptr<vizdoom::DoomGame> game;
  std::unordered_map<Action, std::vector<double>> actionMap;
  std::unordered_map<vizdoom::GameVariable, std::shared_ptr<DoomRewardVariable>> rewardVars;

  // Configuration Variables
  size_t frameSkip;

  bool isEpisodeFinished() const;

  void initGameOptions();
  void initGameActions();
  void initGameVariables();
};

inline DoomEnv::~DoomEnv() { this->game->close(); }

// Start a new episode of play/training
inline void DoomEnv::reset() {
  this->game->newEpisode();
  // (Re)initialize all our variables (used to track rewards)
  for (auto& [varType, varPtr] : this->rewardVars) { varPtr->reinit(*this->game); }
}

/**
 * Execute a single action and step forward to the next state.
 * @returns An std::tuple of the form <nextState:State, reward:double, epsiodeFinished:bool>
 */
inline DoomEnv::StepInfo DoomEnv::step(const DoomEnv::Action& a) {
  auto reward = 0.0;
  auto gameActionVec = this->actionMap[a];

  // Calculate the current action and total state reward based on our reward variables
  // NOTE: You can make a "prolonged" action and skip frames by providing the 2nd arg to makeAction
  reward += this->game->makeAction(gameActionVec, this->frameSkip); // This will advance the ViZDoom game state

  // TODO?
  // std::vector<ImageBufferPtr> stateBuffers;
  // stateBuffers.reserve(this->frameSkip);
  // for (auto i = 0; i < this->frameSkip; i++) { 
  //  reward += this->game->makeAction(gameActionVec, 1);
  //  auto gameState = this->game->getState();
  //  assert(gameState != nullptr);
  //  stateBuffers.push_back(gameState->screenBuffer);
  // }
  auto gameState = this->game->getState();
  assert(gameState != nullptr);

  for (auto& [varType, varPtr] : this->rewardVars) { 
    reward += varPtr->updateAndCalculateReward(*this->game);
  }
  // If the map end is reached the agent gets a big fat reward
  if (this->game->isMapEnded()) { // NOTE: isMapEnded was added to the DoomGame class manually and ViZDoom was recompiled with it!
    reward += 100.0;
  }

  return std::make_tuple(std::make_unique<DoomEnv::State>(gameState->screenBuffer), reward, this->isEpisodeFinished());
}


#endif // __DOOMENV_HPP__