
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "RNG.hpp"
#include "DoomEnv.hpp"
#include "DoomGuy.hpp"
#include "DoomRLLogger.hpp"

namespace fs = std::filesystem;

constexpr char logDirPath[] = "./logs";
constexpr char checkptDirBasePath[] = "./checkpoints";

constexpr char stepsPerEpMaxOpt[]     = "-s";
constexpr size_t minStepsPerEpMax     = 500;
constexpr size_t stepsPerEpMaxDefault = 1000;

constexpr char episodesOpt[]     = "-e";
constexpr size_t minNumEpisodes  = 10;
constexpr size_t episodesDefault = 10;

constexpr char stepsExploreOpt[]     = "-x";
constexpr size_t minStepsExplore     = 100;
constexpr size_t stepsExploreDefault = 1000;

char* getCmdOption(char** begin, char** end, const std::string& option) {
  char** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end) {
    return *itr;
  }
  return 0;
}

bool cmdOptionExists(char** begin, char** end, const std::string& option) {
    return std::find(begin, end, option) != end;
}

void printUsage(const char* programName) {
  auto usageSpacing = 20;
  std::cout << "Usage:" << std::endl;
  std::cout << programName << " (-h|-e <number>|-s <number>|-x)" << std::endl;
  std::cout << std::left << std::setw(usageSpacing) << "-h" << "Help/Usage" << std::endl;
  std::cout << std::left << std::setw(usageSpacing) << episodesOpt << " <# episodes>" << "Specify the number of episodes to run. Default=" << episodesDefault << std::endl;
  std::cout << std::left << std::setw(usageSpacing) << stepsPerEpMaxOpt << " <# max steps per episode>" << 
  "Specify the maximum number of steps that occur in each episode. Default=" << stepsPerEpMaxDefault << std::endl;
  std::cout << std::left << std::setw(usageSpacing) << stepsExploreOpt << " <# exploration steps>" << "Specify the number of steps to explore before starting training. Default=" << stepsExploreDefault << std::endl;
}

int main(int argc, char* argv[]) {

  auto numEpisodes   = episodesDefault;
  auto stepsPerEpMax = stepsPerEpMaxDefault;
  auto stepsExplore  = stepsExploreDefault;
  if (argc > 1) {
    if (cmdOptionExists(argv, argv+argc, "help") || cmdOptionExists(argv, argv+argc, "-h")) {
      printUsage(argv[0]);
      return 0;
    }
    if (argc > 2) {
      // Update the number of episodes if the provided arguement is valid
      auto episodeCmdVal = atoi(getCmdOption(argv, argv+argc, episodesOpt));
      if (episodeCmdVal < minNumEpisodes) {
        std::cout << "Invalid episodes (" << episodesOpt << ") specified, must be >= " << minNumEpisodes << "." << std::endl;
        std::cout << "Defaulting to " << episodesDefault << " episodes." << std::endl;
      }
      else {
        numEpisodes = episodeCmdVal;
      }

      // ... max number of steps per episode
      auto numStepsCmdVal = atoi(getCmdOption(argv, argv+argc, stepsPerEpMaxOpt));
      if (numStepsCmdVal < minStepsPerEpMax) {
        std::cout << "Invalid steps per episode (" << stepsPerEpMaxOpt << ") specified, must be >= " << minStepsPerEpMax << "." << std::endl;
        std::cout << "Defaulting to " << stepsPerEpMaxDefault << " steps per episode." << std::endl;
      }
      else {
        stepsPerEpMax = static_cast<size_t>(numStepsCmdVal);
      }

      // ... number of exploration steps
      auto stepsExploreCmdVal = atoi(getCmdOption(argv, argv+argc, stepsExploreOpt));
      if (stepsExploreCmdVal < minStepsExplore || stepsExploreCmdVal >= stepsPerEpMax) {
        std::cout << "Invalid exploration steps (" << stepsExploreOpt << ") specified, must be >= " << minStepsExplore << "and < " << stepsPerEpMax << " (i.e., max steps per episode)." << std::endl;
        std::cout << "Defaulting to " << stepsExploreDefault << " exploration steps." << std::endl;
      }
      else {
        stepsExplore = static_cast<size_t>(stepsExploreCmdVal);
      }
    }
  }

  auto useCuda = torch::cuda::is_available();
  std::cout << "Using Cuda: " << (useCuda ? "yes" : "no") << std::endl;

  // Setup our logger
  fs::create_directories(logDirPath);
  auto logger = std::make_unique<DoomRLLogger>(logDirPath);

  // Setup our checkpoint save directory
  auto currTime = std::time(nullptr);
  std::stringstream checkptDirPathSS;
  checkptDirPathSS << checkptDirBasePath << "/" << std::put_time(std::localtime(&currTime), "%Y-%m-%dT%H-%M-%S");
  const auto checkPtDirPath = checkptDirPathSS.str();
  fs::create_directories(checkPtDirPath);

  // Setup our agent and gym / environment
  auto guy = std::make_unique<DoomGuy>(checkPtDirPath, stepsExplore);
  auto env = std::make_unique<DoomEnv>(stepsPerEpMax);

  try {
    std::cout << "Running " << numEpisodes << " episodes of DoomGuy..." << std::endl;
    for (auto e = 0; e < numEpisodes; e++) {
      const auto currEpNum = e+1;
      
      std::cout << "Starting episode #" << currEpNum << "..." << std::endl;
      std::cout << "Maximum steps expected: " << stepsPerEpMax << std::endl;

      // Time to play Doom!
      auto state = env->reset(); // Reset the environment and get the initial state
      while (true) {

        auto action = guy->act(state); // Run DoomGuy on the current state

        // Perform the action in the environment and observe the next state, 
        // reward and whether we're finished the episode
        auto [nextState, reward, done] = env->step(action);

        // Remember the full step sequence for recall later on
        guy->cache(state, nextState, action, reward, done);

        // Learn - this may just exit with [-1,-1] if we're still exploring
        auto [q, loss] = guy->learn();
        
        logger->logStep(reward, loss, q); // Log our results for the step
        
        if (env->getStepsPerformed() % 500 == 0) {
          std::cout << "[Episode #" << currEpNum << "]: " << env->getStepsPerformed() << " steps performed so far..." << std::endl;
        }

        // Update to the next state and check to see if we're done the episode
        if (done) {
          std::cout << "Finished episode." << std::endl;
          break;
        }
        state = std::move(nextState);
      }

      // Perform episode and running average logging
      logger->logEpisode();
      //if (e % minNumEpisodes == 0) {
      logger->record(currEpNum, guy->getExplorationRate(), guy->getCurrStep());
      //}
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception thrown: " << std::endl << e.what() << std::endl << "Terminating program." << std::endl;
  }

  return 0;
}