#include <iostream>
#include <cstdlib>
#include <sstream>
#include <memory>
#include <chrono>

#include <torch/torch.h>

#include "utils/StringUtils.hpp"
#include "utils/RNG.hpp"

#include "lr_schedulers/ConstantLRScheduler.hpp"
#include "lr_schedulers/CosAnnealLRScheduler.hpp"

#include "debug_doom_rl.hpp"
#include "DoomEnv.hpp"
#include "DoomGuy.hpp"
#include "DoomRLLogger.hpp"
#include "DoomRLCmdOpts.hpp"
#include "ReplayMemory.hpp"


constexpr size_t actionFrameSkip = 4;
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

  // Setup agent and gym / environment, etc.
  //auto lrScheduler  = cmdOpts->isLearningRateConstant() ? std::make_unique()
  auto guy = std::make_unique<DoomGuy>(checkPtDirPath, cmdOpts);
  auto env = std::make_unique<DoomEnv>(cmdOpts->stepsPerEpMax, actionFrameSkip, cmdOpts->isActivePlay);

  // If a checkpoint file was given, load it
  if (!cmdOpts->checkpointFilepath.empty()) {
    guy->load(cmdOpts->checkpointFilepath);
  }
  logger->logStartSession(*cmdOpts, *guy);

  // Setup our replayMemory, make sure we do this AFTER loading the checkpoint!
  auto [stateHeight,stateWidth] = DoomEnv::State::getNetInputSize(guy->getNetworkVersion());
  auto replayMemory = std::make_unique<ReplayMemory>(stateHeight, stateWidth, DoomEnv::State::NUM_CHANNELS);

  // Check for cycling maps
  std::vector<std::string> cycleMaps;
  if (cmdOpts->doomMap.find(',') != std::string::npos) {
    cycleMaps = StringUtils::split(cmdOpts->doomMap, ',');
  }

  guy->train(true);

  auto prevTime = std::chrono::steady_clock::now();
  std::cout << "Running " << cmdOpts->numEpisodes << " episodes of DoomGuy..." << std::endl;

  for (auto e = 0; e < cmdOpts->numEpisodes; e++) {
    const auto currEpNum = e+1;

    auto updateEnvMap = [&env, &cmdOpts, cycleMaps, e]() {
      if (cmdOpts->doomMap.compare(DoomRLCmdOpts::doomMapCycle) == 0) {
        if (e != 0) { env->setCycledMap(); }
      }
      else if (cmdOpts->doomMap.compare(DoomRLCmdOpts::doomMapRandom) == 0) {
        env->setRandomMap();
      }
      else if (cycleMaps.size() > 0) {
        static auto currCycleMapIdx = 0;
        env->setMap(cycleMaps[currCycleMapIdx]);
        currCycleMapIdx = (currCycleMapIdx+1) % cycleMaps.size();
      }
      else {
        env->setMap(cmdOpts->doomMap);
      }
    };
    
    std::cout << "Starting episode #" << currEpNum << "..." << std::endl;

    // Time to play Doom!
    // Set the map, reset the environment and get the initial state
    updateEnvMap();
    
    const auto networkVersion = guy->getNetworkVersion();
    auto state = env->reset(networkVersion);
    replayMemory->initState(state);

    while (true) {
      auto action = guy->act(state, env); // Run DoomGuy on the current state

      // Perform the action in the environment and observe the next state, 
      // reward and whether we're finished the episode
      auto [nextState, reward, done] = env->step(action, networkVersion);

      // Remember the full step sequence for recall later on
      replayMemory->cache(nextState, static_cast<int>(action), reward, done);
      if (replayMemory->getCacheSize() < ReplayMemory::DEFAULT_REPLAY_BATCH_SIZE+1) { continue; }

      // Learn - this may just exit with [-1,-1] if we're still exploring
      auto [q, loss] = guy->learn(env, replayMemory, ReplayMemory::DEFAULT_REPLAY_BATCH_SIZE);
      
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

    logger->logEpisode(currEpNum, guy->getCurrStep(), env->getMapToLoadNext());
    guy->episodeEnded(currEpNum);
  }

  return 0;
}