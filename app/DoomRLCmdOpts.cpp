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
constexpr char cmdEpislonMin[]    = "epsilon_min";
constexpr char cmdEpsilonDecay[]  = "epsilon_decay";
constexpr char cmdLoadCkpt[]      = "checkpoint";
constexpr char cmdMap[]           = "map";

constexpr char cmdLearningRate[]     = "lr";
constexpr char cmdMinLearningRate[]  = "min_lr";
constexpr char cmdMaxLearningRate[]  = "max_lr";

constexpr char cmdIsExecTesting[]    = "training_off";

//constexpr char cmdActivePlay[]    = "active_play";

constexpr size_t stepsPerEpMaxDefault = 1e5;
constexpr size_t episodesDefault      = 1e5;
constexpr size_t stepsExploreDefault  = 1e4;
constexpr size_t stepsSaveDefault     = 5e5;
constexpr size_t stepsSyncDefault     = 1e4;
constexpr double epislonDefault       = 1.0;
constexpr double epsilonMinDefault    = 0.1;
constexpr double epsilonDecayDefault  = 0.9999998;
constexpr double learningRateDefault  = 2.5e-4;
constexpr bool isActivePlayDefault    = false;

DoomRLCmdOpts::DoomRLCmdOpts(int argc, char* argv[]): desc("Allowed options") {
  this->desc.add_options()
    (cmdHelp, "Print help/usage message.")
    (cmdEpisodes,         po::value<size_t>(&this->numEpisodes)->default_value(episodesDefault),         "Number of episodes to run.")
    (cmdStepsExplore,     po::value<size_t>(&this->stepsExplore)->default_value(stepsExploreDefault),    "Number of steps to explore before starting training.")
    (cmdStepsSave,        po::value<size_t>(&this->stepsSave)->default_value(stepsSaveDefault),          "Number of steps between checkpoints (i.e., when the network model is saved to disk).")
    (cmdStepsSync,        po::value<size_t>(&this->stepsSync)->default_value(stepsSyncDefault),          "Number of steps between when the Q-target network is synchronized with the Q-online network.")
    (cmdEpsilon,          po::value<double>(&this->startEpsilon)->default_value(epislonDefault),         "Starting epsilon value [1,0]. This is how likely the agent is to explore (choose a random action, where 1 is always explore).")
    (cmdEpsilonDecay,     po::value<double>(&this->epsilonDecay)->default_value(epsilonDecayDefault),    "Epsilon decay multiplier per step (multiplies the epsilon value at each step, decaying it over the course of training.")
    (cmdLearningRate,     po::value<double>(&this->learningRate)->default_value(learningRateDefault),    "Initial learning rate for gradient descent / optimization of the model network. Note: If the minimum and/or maximum bounds on learning rate are not set then the rate will be constant throughout training.")
    (cmdMinLearningRate,  po::value<double>(&this->minLearningRate)->default_value(learningRateDefault), "The minimum bound on the learning rate.")
    (cmdMaxLearningRate,  po::value<double>(&this->maxLearningRate)->default_value(learningRateDefault), "The maximum bound on the learning rate.")
    (cmdLoadCkpt,         po::value<std::string>(&this->checkpointFilepath)->default_value(""),          "Filepath for loading a checkpoint model file for the Q-networks.")
    (cmdMap,              po::value<std::string>(&this->doomMap)->default_value("E1M1"),                 "Doom map to train in, to cycle through maps use 'cycle', to choose random maps use 'random', to cycle through a list of maps enter a set of comma separated map names.")
    //(cmdActivePlay,       po::bool_switch(&this->isActivePlay)->default_value(isActivePlayDefault),     "Whether the player (i.e., you) will play in place of the agent for teaching purposes.")
    (cmdEpislonMin,       po::value<double>(&this->epsilonMin)->default_value(epsilonMinDefault),        "The minimum allowable epsilon value (used in epsilon-greedy policy) during training.")
    (cmdIsExecTesting,    po::bool_switch(&this->isExecTesting)->default_value(false),                       "If this flag is provided then training will be turned off and the agent will play Doom.")
  ;
  po::store(po::parse_command_line(argc, argv, desc), this->vm);
  po::notify(this->vm);

  this->checkOpts();
}

void DoomRLCmdOpts::printOpts(std::ostream& stream) const {
    auto logVarInfo = [&stream](const auto& preamble, auto value) {
      stream << std::left << std::setw(40) << preamble << " " << value << std::endl;
    };
    if (this->isExecTesting) {
      stream << "Training turned off, agent will just be playing Doom." << std::endl;
    }
    else {
      logVarInfo("Number of episodes set to", this->numEpisodes);
      logVarInfo("Exploration steps set to", this->stepsExplore);
      logVarInfo("Steps between saves set to", this->stepsSave);
      logVarInfo("Steps between network sync set to", this->stepsSync);
      logVarInfo("Starting epsilon set to", this->startEpsilon);
      logVarInfo("Epsilon min set to", this->epsilonMin);
      logVarInfo("Starting epsilon decay multiplier set to", this->epsilonDecay);
      logVarInfo("Current learning rate set to", this->learningRate);
      if (this->isLearningRateConstant()) {
        std::cout << "*** The learning rate will be constant throughout training ***" << std::endl;
      }
      else {
        std::cout << "*** The learning rate will be bounded in the interval [" 
                  << std::fixed << std::setprecision(8) << this->minLearningRate << ", " << this->maxLearningRate << "] ***" << std::endl;
      }
    }
    logVarInfo("Checkpoint file set to", this->checkpointFilepath.empty() ? "<empty>" : this->checkpointFilepath);
    logVarInfo("Doom map set to", this->doomMap);
}

void DoomRLCmdOpts::checkOpts() {
  if (this->vm.count(cmdHelp)) {
    std::cout << this->desc << std::endl;
    exit(0);
  }

  this->cmdVarCheck<size_t>(cmdEpisodes, this->numEpisodes, episodesDefault, "number of episodes", 1, std::numeric_limits<size_t>::max());
  this->cmdVarCheck<size_t>(cmdStepsExplore, this->stepsExplore, stepsExploreDefault, "exploration steps", 32, std::numeric_limits<size_t>::max());
  this->cmdVarCheck<size_t>(cmdStepsSave, this->stepsSave, stepsSaveDefault, "save steps (steps between checkpoints)", 1000, std::numeric_limits<size_t>::max());
  this->cmdVarCheck<size_t>(cmdStepsSync, this->stepsSync, stepsSyncDefault, "steps between synchronization between Q-target and Q-online networks", 1e2, 1e4);
  this->cmdVarCheck<double>(cmdEpsilon, this->startEpsilon, epislonDefault, "starting epsilon", 0.0, 1.0);
  this->cmdVarCheck<double>(cmdEpsilon, this->epsilonMin, epsilonMinDefault, "minimum allowable epsilon", 0.0, 1.0);
  this->cmdVarCheck<double>(cmdEpsilonDecay, this->epsilonDecay, epsilonDecayDefault, "epsilon decay multiplier", 0, 1, false, false);
  this->cmdVarCheck<double>(cmdLearningRate, this->learningRate, learningRateDefault, "learning rate", 0, 20, true, true);

  // Make sure the learning rate bounds make sense
  this->minLearningRate = std::min<double>(this->minLearningRate, this->learningRate);
  this->maxLearningRate = std::max<double>(this->maxLearningRate, this->learningRate);

  if (this->vm.count(cmdLoadCkpt) && !this->checkpointFilepath.empty()) {
    // Check if the file exists...
    if (!std::filesystem::exists(this->checkpointFilepath)) {
      std::cout << "Invalid checkpoint, file specified at '" << this->checkpointFilepath << "' does not exist." << std::endl;
      this->checkpointFilepath = "";
    }
  }
  if (this->isExecTesting) { this->startEpsilon = 0; }
}

template <typename T> 
void DoomRLCmdOpts::cmdVarCheck(const char* cmd, T& var, T defaultVal, const std::string& desc, T min, T max, bool minInc, bool maxInc) {
  if ((minInc && var < min || !minInc && var <= min || maxInc && var > max || !maxInc && var >= max)) {
    std::cout << "Invalid " << desc << " specified, must be in " << (minInc ? "[" : "(") << min << ", " << max << (maxInc ? "]" : ")") << "." << std::endl;
    std::cout << "Defaulting to " << defaultVal << " " << desc << "." << std::endl;
    var = defaultVal;
  }
}