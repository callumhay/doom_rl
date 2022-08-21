#include <iostream>
#include <cstdlib>
#include <sstream>
#include <memory>
#include <chrono>

#include <torch/torch.h>

#include "RNG.hpp"
#include "DoomEnv.hpp"
#include "DoomGuy.hpp"
#include "DoomRLLogger.hpp"
#include "DoomRLCmdOpts.hpp"

constexpr char checkptDirBasePath[] = "./checkpoints";

int main(int argc, char* argv[]) {
  auto cmdOpts = std::make_unique<DoomRLCmdOpts>(argc, argv);
  cmdOpts->printOpts(std::cout);

  auto useCuda = torch::cuda::is_available();
  std::cout << "Using Cuda: " << (useCuda ? "yes" : "no") << std::endl;

  // Setup our checkpoint save directory name
  auto currTime = std::time(nullptr);
  std::stringstream checkptDirPathSS;
  checkptDirPathSS << checkptDirBasePath << "/" << std::put_time(std::localtime(&currTime), "%Y-%m-%dT%H-%M-%S");
  const auto checkPtDirPath = checkptDirPathSS.str();

  // Setup our logger
  auto logger = std::make_unique<DoomRLLogger>(checkptDirBasePath, checkPtDirPath);
  logger->logStartSession(*cmdOpts);

  // Setup our agent and gym / environment
  auto guy = std::make_unique<DoomGuy>(
    checkPtDirPath, cmdOpts->stepsPerEpMax, 
    cmdOpts->stepsExplore, cmdOpts->stepsSave, cmdOpts->stepsSync,
    cmdOpts->startEpsilon, cmdOpts->epsilonDecay, cmdOpts->learningRate
  );
  auto env = std::make_unique<DoomEnv>(cmdOpts->stepsPerEpMax);

  // If a checkpoint file was given, load it
  if (!cmdOpts->checkpointFilepath.empty()) {
    std::cout << "Loading from checkpoint file " << cmdOpts->checkpointFilepath << "..." << std::endl;
    guy->load(cmdOpts->checkpointFilepath);
  }

  auto prevTime = std::chrono::steady_clock::now();

  std::cout << "Running " << cmdOpts->numEpisodes << " episodes of DoomGuy..." << std::endl;
  for (auto e = 0; e < cmdOpts->numEpisodes; e++) {
    const auto currEpNum = e+1;
    
    std::cout << "Starting episode #" << currEpNum << "..." << std::endl;

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
      
      logger->logStep(reward, loss, q, guy->getLearningRate(), guy->getEpsilon()); // Log our results for the step
      
      auto currTime = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::seconds>(currTime-prevTime).count() >= 30) {
        prevTime = currTime;
        std::cout << "[Episode #" << currEpNum << "]: " << env->getStepsPerformed() << " steps of episode, total steps across all episodes: " << guy->getCurrStep() << std::endl;
      }

      // Update to the next state and check to see if we're done the episode
      if (done) {
        std::cout << "Finished episode." << std::endl;
        break;
      }
      state = std::move(nextState);
    }

    logger->logEpisode(currEpNum, guy->getCurrStep());
  }

  return 0;
}