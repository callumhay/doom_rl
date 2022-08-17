
#include <iostream>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "DoomEnv.hpp"
#include "DoomGuy.hpp"
#include "DoomRLLogger.hpp"

namespace fs = std::filesystem;

constexpr char logDirPath[] = "./logs";
constexpr char checkptDirBasePath[] = "./checkpoints";

constexpr char episodesOpt[] = "-e";
constexpr size_t minNumEpisodes  = 10;
constexpr size_t episodesDefault = 10;

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
  std::cout << programName << " (-h|-e <number>)" << std::endl;
  std::cout << std::left << std::setw(usageSpacing) << "-h" << "Help/Usage" << std::endl;
  std::cout << std::left << std::setw(usageSpacing) << "-e <# episodes>" << "Specify the number of episodes to run." << std::endl;
}

int main(int argc, char* argv[]) {
  std::srand(42);

  auto numEpisodes = episodesDefault;
  if (argc > 1) {
    if (cmdOptionExists(argv, argv+argc, "help") || cmdOptionExists(argv, argv+argc, "-h")) {
      printUsage(argv[0]);
      return 0;
    }
    if (argc > 2) {
      // Update the number of episodes if the provided arguement is valid
      auto value = atoi(getCmdOption(argv, argv+argc, episodesOpt));
      if (value < 10) {
        std::cout << "Invalid episodes (" << episodesOpt << ") specified, must be >= 10." << std::endl;
        std::cout << "Defaulting to 10 episodes." << std::endl;
      }
      else {
        numEpisodes = value;
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
  auto guy = std::make_unique<DoomGuy>(checkPtDirPath);
  auto env = std::make_unique<DoomEnv>();

  std::cout << "Running " << numEpisodes << " episodes of DoomGuy..." << std::endl;
  for (auto e = 0; e < numEpisodes; e++) {
    
    std::cout << "Starting episode #" << (e+1) << "..." << std::endl;

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

      // Update to the next state and check to see if we're done the episode
      state = std::move(nextState);
      if (done) {
        std::cout << "Finished episode." << std::endl;
        break;
      }
    }

    // Perform episode and running average logging
    logger->logEpisode();
    if (e % minNumEpisodes == 0) {
      logger->record(e, guy->getExplorationRate(), guy->getCurrStep());
    }
  }


  /*
  std::vector<uint8_t> fb = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
  auto opts = torch::TensorOptions().dtype(torch::kUInt8);
  auto fbTensor = torch::from_blob(fb.data(), {4,4}, opts).clone();
  std::cout << fbTensor << std::endl;
  // Grab the center 2x2 of fbTensor
  using namespace torch::indexing;
  auto fbCenter2x2 = fbTensor.index({Slice(1,3), Slice(1,3)});
  std::cout << fbCenter2x2 << std::endl;

  std::cout << fbTensor.permute({1,0}).unsqueeze(0) << std::endl;

  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  auto oneValTensor = torch::tensor({1});
  auto item = oneValTensor.item();
  std::cout << typeid(item).name() << std::endl;
  */
  return 0;
}