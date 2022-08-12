#include "ViZDoom.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>

#include "DoomEnvironment.hpp"

template<typename T> class TD; 

using namespace vizdoom;

int main() {
  std::srand(time(0));
  const int episodes = 10; // Run this many episodes

  std::cout << "STARTING VIZDOOM ENVIRONMENT..." << std::endl;

  DoomEnvironment env(1000);

  // Sets time that will pause the engine after each action.
  // Without this everything would go too fast for you to keep track of what's happening.
  unsigned int sleepTime = 1000 / DEFAULT_TICRATE; // = 28

  for (auto i = 0; i < episodes; i++) {
    std::cout << "Starting Episode #" << i + 1 << std::endl;

    auto currState = env.InitialSample();
    DoomEnvironment::State nextState(nullptr);
    auto totalEpisodeReward = 0.0;

    while (!env.IsTerminal(currState)) {
      DoomEnvironment::Action action;
      action.action = static_cast<DoomEnvironment::Action::actions>(std::rand() % DoomEnvironment::Action::size);
      auto reward = env.Sample(currState, action, nextState);
      currState = nextState;

      if (reward != 0) {
        std::cout << "State #" << env.StepsPerformed() << std::endl;
        std::cout << "Reward: " << reward << std::endl;
        std::cout << "=====================" << std::endl;
        totalEpisodeReward += reward;
      }

      // TODO: Remove this...
      std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
    }

    std::cout << "Episode finished." << std::endl;
    std::cout << "Total reward: " << totalEpisodeReward << std::endl;
    std::cout << "************************" << std::endl;
  }
}