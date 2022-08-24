#include <iostream>
#include <filesystem>

#include "DoomRLCmdOpts.hpp"

namespace po = boost::program_options;

constexpr char cmdHelp[]          = "help";
constexpr char cmdEpisodes[]      = "episodes";
constexpr char cmdStepsPerEpMax[] = "episode_steps";
constexpr char cmdStepsExplore[]  = "explore_steps";
constexpr char cmdStepsSave[]     = "save_steps";
constexpr char cmdStepsSync[]     = "sync_steps";
constexpr char cmdEpsilon[]       = "epsilon";
constexpr char cmdEpsilonDecay[]  = "epsilon_decay";
constexpr char cmdLearningRate[]  = "lr";
constexpr char cmdLoadCkpt[]      = "checkpoint";
constexpr char cmdMap[]           = "map";

constexpr size_t stepsPerEpMaxDefault = 1e5;
constexpr size_t episodesDefault      = 1e5;
constexpr size_t stepsExploreDefault  = 1e4;
constexpr size_t stepsSaveDefault     = 5e5;
constexpr size_t stepsSyncDefault     = 1e4;
constexpr double epislonDefault       = 1.0;
constexpr double epsilonDecayDefault  = 0.99999975;
constexpr double learningRateDefault  = 2.5e-4;

DoomRLCmdOpts::DoomRLCmdOpts(int argc, char* argv[]): desc("Allowed options") {
  this->desc.add_options()
    (cmdHelp, "Print help/usage message.")
    (cmdEpisodes,      po::value<size_t>(&this->numEpisodes)->default_value(episodesDefault),        "Number of episodes to run.")
    (cmdStepsPerEpMax, po::value<size_t>(&this->stepsPerEpMax)->default_value(stepsPerEpMaxDefault), "Maximum number of steps per episode.")
    (cmdStepsExplore,  po::value<size_t>(&this->stepsExplore)->default_value(stepsExploreDefault),   "Number of steps to explore before starting training.")
    (cmdStepsSave,     po::value<size_t>(&this->stepsSave)->default_value(stepsSaveDefault),         "Number of steps between checkpoints (i.e., when the network model is saved to disk).")
    (cmdStepsSync,     po::value<size_t>(&this->stepsSync)->default_value(stepsSyncDefault),         "Number of steps between when the Q-target network is synchronized with the Q-online network.")
    (cmdEpsilon,       po::value<double>(&this->startEpsilon)->default_value(epislonDefault),        "Starting epsilon value [1,0]. This is how likely the agent is to explore (choose a random action, where 1 is always explore).")
    (cmdEpsilonDecay,  po::value<double>(&this->epsilonDecay)->default_value(epsilonDecayDefault),   "Epsilon decay multiplier per step (multiplies the epsilon value at each step, decaying it over the course of training.")
    (cmdLearningRate,  po::value<double>(&this->learningRate)->default_value(learningRateDefault),   "Initial learning rate for gradient descent / optimization of the model network.")
    (cmdLoadCkpt,      po::value<std::string>(&this->checkpointFilepath)->default_value(""),         "Filepath for loading a checkpoint model file for the Q-networks.")
    (cmdMap,           po::value<std::string>(&this->doomMap)->default_value("E1M1"),                "Doom map to train in, to cycle through maps use 'cycle', to choose random maps use 'random'.")
  ;
  po::store(po::parse_command_line(argc, argv, desc), this->vm);
  po::notify(this->vm);

  this->checkOpts();
}

void DoomRLCmdOpts::printOpts(std::ostream& stream) const {
    auto logVarInfo = [&stream](const auto& preamble, auto value) {
      stream << std::left << std::setw(40) << preamble << " " << value << std::endl;
    };
    logVarInfo("Number of episodes set to", this->numEpisodes);
    logVarInfo("Maximum steps per episode set to", this->stepsPerEpMax);
    logVarInfo("Exploration steps set to", this->stepsExplore);
    logVarInfo("Steps between saves set to", this->stepsSave);
    logVarInfo("Steps between network sync set to", this->stepsSync);
    logVarInfo("Starting epsilon set to", this->startEpsilon);
    logVarInfo("Starting epsilon decay multiplier set to", this->epsilonDecay);
    logVarInfo("Starting learning rate set to", this->learningRate);
    logVarInfo("Checkpoint file set to", this->checkpointFilepath);
    logVarInfo("Doom map set to", this->doomMap);
}

void DoomRLCmdOpts::checkOpts() {
  if (this->vm.count(cmdHelp)) {
    std::cout << this->desc << std::endl;
    exit(0);
  }

  this->cmdVarCheck<size_t>(cmdEpisodes, this->numEpisodes, episodesDefault, "number of episodes", 1, std::numeric_limits<size_t>::max());
  this->cmdVarCheck<size_t>(cmdStepsPerEpMax, this->stepsPerEpMax, stepsPerEpMaxDefault, "steps per episode", 500, std::numeric_limits<size_t>::max());
  this->cmdVarCheck<size_t>(cmdStepsExplore, this->stepsExplore, stepsExploreDefault, "exploration steps", 32, std::numeric_limits<size_t>::max());
  this->cmdVarCheck<size_t>(cmdStepsSave, this->stepsSave, stepsSaveDefault, "save steps (steps between checkpoints)", 1000, std::numeric_limits<size_t>::max());
  this->cmdVarCheck<size_t>(cmdStepsSync, this->stepsSync, stepsSyncDefault, "steps between synchronization between Q-target and Q-online networks", 1e2, 1e4);
  this->cmdVarCheck<double>(cmdEpsilon, this->startEpsilon, epislonDefault, "starting epsilon", 0.0, 1.0);
  this->cmdVarCheck<double>(cmdEpsilonDecay, this->epsilonDecay, epsilonDecayDefault, "epsilon decay multiplier", 0, 1, false, false);
  this->cmdVarCheck<double>(cmdLearningRate, this->learningRate, learningRateDefault, "learning rate", 0, 1, true, false);

  if (this->vm.count(cmdLoadCkpt) && !this->checkpointFilepath.empty()) {
    // Check if the file exists...
    if (!std::filesystem::exists(this->checkpointFilepath)) {
      std::cout << "Invalid checkpoint, file specified at '" << this->checkpointFilepath << "' does not exist." << std::endl;
      this->checkpointFilepath = "";
    }
  }
}

template <typename T> 
void DoomRLCmdOpts::cmdVarCheck(const char* cmd, T& var, T defaultVal, const std::string& desc, T min, T max, bool minInc, bool maxInc) {
  if ((minInc && var < min || !minInc && var <= min || maxInc && var > max || !maxInc && var >= max)) {
    std::cout << "Invalid " << desc << " specified, must be in " << (minInc ? "[" : "(") << min << ", " << max << (maxInc ? "]" : ")") << "." << std::endl;
    std::cout << "Defaulting to " << defaultVal << " " << desc << "." << std::endl;
    var = defaultVal;
  }
}