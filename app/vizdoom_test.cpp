#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <vector>

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/reinforcement_learning/q_learning.hpp>
#include <mlpack/methods/reinforcement_learning/q_networks/simple_dqn.hpp>
#include <mlpack/methods/reinforcement_learning/policy/greedy_policy.hpp>
#include <mlpack/methods/reinforcement_learning/training_config.hpp>

#include "ViZDoom.h"

#include "DoomEnvironment.hpp"

//template<typename T> class TD; 

using namespace mlpack;
using namespace mlpack::ann;
using namespace mlpack::rl;
using namespace ens;
using namespace vizdoom;

const int MAX_STEPS_PER_EPISODE = 128;

int main() {
  std::srand(time(0));

  // Setup our NN for calculating the Q function (via deep Q-Learning) in RL
  FFN<MeanSquaredError, GaussianInitialization> network(MeanSquaredError(), GaussianInitialization(0, 1));
  
  network.InputDimensions() = std::vector<size_t>({
    DoomEnvironment::State::INPUT_WIDTH, DoomEnvironment::State::INPUT_HEIGHT, DoomEnvironment::State::NUM_CHANNELS
  });
  /*
  // [W, H, D] = [320, 200, 3]
  const auto filterSize0   = 5;
  const auto filterStride0 = 3;
  network.Add<Convolution>(
    8,  // Number of output activation maps
    filterSize0,  // Filter width.
    filterSize0,  // Filter height.
    filterStride0,  // Stride along width.
    filterStride0,  // Stride along height.
    0,  // Padding width.
    0  // Padding height.
  );
  network.Add<LeakyReLU>();
  network.Add<MaxPooling>(
    2, // Width of field.
    2, // Height of field.
    2, // Stride along width.
    2, // Stride along height.
    true // Rounding operator is 'floor', not 'ceil'
  );

  const auto filterSize1 = 3;
  const auto filterStride1 = 2;
  network.Add<Convolution>(
    8, // Output maps (number of filters)
    filterSize1,  // Filter width
    filterSize1,  // Filter height
    filterStride1,  // Stride (width)
    filterStride1,  // Stride (height)
    0,  // Padding (width)
    0   // Padding (height)
  );
  network.Add<LeakyReLU>();
  network.Add<MaxPooling>(2, 2, 2, 2, true);

  const auto filterSize2 = 2;
  const auto filterStride2 = 2;
  network.Add<Convolution>(
    8, // Output maps (number of filters)
    filterSize2,  // Filter width
    filterSize2,  // Filter height
    filterStride2,  // Stride (width)
    filterStride2,  // Stride (height)
    0,  // Padding (width)
    0   // Padding (height)
  );
  network.Add<LeakyReLU>();
  network.Add<MaxPooling>(2, 2, 2, 2, true);
  */

  // Start off with a small-ish network...
  // [W, H, D] = [320, 200, 3]
  const auto filterSize0   = 9;
  const auto filterStride0 = 4;
  network.Add<Convolution>(
    8,  // Number of output activation maps
    filterSize0,  // Filter width.
    filterSize0,  // Filter height.
    filterStride0,  // Stride along width.
    filterStride0,  // Stride along height.
    0,  // Padding width.
    0  // Padding height.
  );
  network.Add<LeakyReLU>();
  network.Add<MaxPooling>(
    3, // Width of field.
    3, // Height of field.
    3, // Stride along width.
    3, // Stride along height.
    true // Rounding operator is 'floor', not 'ceil'
  );

  // Densely connected final 2 layers
  network.Add<Linear>(64);
  network.Add<ReLU>();
  network.Add<Linear>(DoomEnvironment::Action::size);

  SimpleDQN model(network);

  GreedyPolicy<DoomEnvironment> policy(
    1.0,  // Initial epsilon (likelihood of choosing a random action)
    MAX_STEPS_PER_EPISODE, // The steps during which the probability (epsilon) to explore will decay.
    0.1,  // Epsilon will never go less than this
    0.99  // Rate at which epsilon will decrease
  );
  RandomReplay<DoomEnvironment> replayMethod(
    32, // Number of examples returned at each sample.
    512 // Total number of state examples to store
  );

  // Set up training configurations.
  TrainingConfig config;
  config.TargetNetworkSyncInterval() = 128; // Interval for syncing with the target network.
  config.ExplorationSteps()          = MAX_STEPS_PER_EPISODE-1; // The agent won't start to learn until this number of steps have passed.
  config.StepLimit()                 = MAX_STEPS_PER_EPISODE; // ???
  config.NumWorkers()                = std::thread::hardware_concurrency();

  // Set up DQN agent.
  auto doomGame = new DoomGame();
  auto agent = QLearning<DoomEnvironment, decltype(model), AdamUpdate, decltype(policy), decltype(replayMethod)>(
    config, model, policy, replayMethod, AdamUpdate(), DoomEnvironment(doomGame, MAX_STEPS_PER_EPISODE)
  );

  // Training *******************************************
  const size_t numTrainingEpisodes = 10;
  arma::running_stat<double> averageReturn;
  size_t episodes = 0;

  agent.Deterministic() = false; // Needed for training!
  std::cout << "Training for " << numTrainingEpisodes << " episodes." << std::endl;
  while (episodes < numTrainingEpisodes) {
    std::cout << "--------------------------" << std::endl;
    std::cout << "Starting Episode #" << (episodes+1) << std::endl;
    
    double episodeReturn = agent.Episode();
    averageReturn(episodeReturn);
    episodes++;

    std::cout << "Episode #" << episodes << " Completed." << std::endl;
    std::cout << "Episode return: " << episodeReturn << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << "Average return across episodes: " << averageReturn.mean() << std::endl;
  }
  std::cout << "Training completed!" << std::endl;
  // End of Training *************************************

  /*
  std::cout << "STARTING VIZDOOM ENVIRONMENT..." << std::endl;
  const int episodes = 10; // Run this many episodes
  const size_t actionsPerSecond = 4;
  const auto frameSkip = DEFAULT_TICRATE/actionsPerSecond;
  auto env = std::make_shared<DoomEnvironment>();
  env->MaxSteps()  = MAX_STEPS_PER_EPISODE;
  env->FrameSkip() = frameSkip;

  // Sets time that will pause the engine after each action.
  // Without this everything would go too fast for you to keep track of what's happening.
  unsigned int sleepTime = (1000 / DEFAULT_TICRATE) * frameSkip;

  for (auto i = 0; i < episodes; i++) {
    std::cout << "Starting Episode #" << i + 1 << std::endl;

    auto currState = env->InitialSample();
    DoomEnvironment::State nextState(nullptr);
    auto totalEpisodeReward = 0.0;

    do {
      DoomEnvironment::Action action;
      action.action = static_cast<DoomEnvironment::Action::Actions>(std::rand() % DoomEnvironment::Action::size);
      auto reward = env->Sample(currState, action, nextState);
      currState = nextState;

      if (reward != 0) {
        std::cout << "State #" << env->StepsPerformed() << std::endl;
        std::cout << "Reward: " << reward << std::endl;
        std::cout << "=====================" << std::endl;
        totalEpisodeReward += reward;
      }

      // TODO: Remove this...
      std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
    } while (!env->IsTerminal(currState));

    std::cout << "Episode finished." << std::endl;
    std::cout << "Total reward: " << totalEpisodeReward << std::endl;
    std::cout << "************************" << std::endl;
  }
  */
  doomGame->close();
  delete doomGame;
}