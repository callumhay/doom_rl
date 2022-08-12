#include <utility>

#include "ViZDoom.h"
#include "DoomEnvironment.hpp"

using namespace vizdoom;

typedef DoomEnvironment::Action::actions ActionType;
const size_t EPISODE_START_TICKS = 10;
const size_t TICKS_PER_ACTION    = 1;

DoomEnvironment::DoomEnvironment(size_t maxSteps) : maxSteps(maxSteps), stepsPerformed(0), game(new DoomGame()) {
  // Setup the ViZDoom game environment...
  this->initGameOptions();
  this->initGameActions();
  this->initGameVariables();

  // Special options, specific to the training scenario:
  // ----------------------------------------------------
  // Set map to start (scenario .wad files can contain many maps).
  this->game->setDoomMap("E1M1");
  // Causes episodes to finish after the given number of ticks
  this->game->setEpisodeTimeout(maxSteps*TICKS_PER_ACTION + EPISODE_START_TICKS);
  // Makes episodes start after 10 tics (~after raising the weapon)
  this->game->setEpisodeStartTime(EPISODE_START_TICKS);
  // Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
  this->game->setMode(PLAYER);

  // Initialize the game. Further configuration won't take any effect from now on.
  this->game->init();
}

/**
 * Dynamics of playing Doom. Get the reward and next state based on the current state and action.
 * @param state The current state.
 * @param action The current action.
 * @param nextState The next state.
 * @return The current, calculated reward for the the given state-action pair.
 */
double DoomEnvironment::Sample(const DoomEnvironment::State& state, const DoomEnvironment::Action& action, DoomEnvironment::State& nextState) {
  // Advance the state and collect the reward (MUST be done before setting the next state!)
  auto reward = this->Sample(state, action);

  // Setup the next state
  auto gameState = this->game->getState();
  nextState.SetScreenBuf(gameState->screenBuffer);

  return reward;
}

double DoomEnvironment::Sample(const DoomEnvironment::State& state, const DoomEnvironment::Action& action) {
  this->stepsPerformed++;
  auto gameActionVec = this->actionMap[action.action];
  
  // Calculate the current action and total state reward based on our reward variables
  // NOTE: You can make a "prolonged" action and skip frames: double reward = game->makeAction(choice(actions), skiprate)
  auto reward = this->game->makeAction(gameActionVec);
  for (auto& [varType, varPtr] : this->rewardVars) { 
    reward += varPtr->updateAndCalculateReward(game);
  }
  // If the map end is reached the agent gets a big fat reward
  if (this->game->isMapEnded()) { // NOTE: isMapEnded was added to the DoomGame class manually and ViZDoom was recompiled with it!
    reward += 100.0;
  }

  return reward;
}

/**
 * Setup and get the initial state for an episode.
 * @return Initial state for each episode.
 */
DoomEnvironment::State DoomEnvironment::InitialSample() {
  // Do everything we need to setup the initial state of the game...
  this->stepsPerformed = 0;
  this->game->newEpisode();
  // (Re)initialize all our variables (used to track rewards)
  for (auto& [varType, varPtr] : this->rewardVars) { varPtr->reinit(this->game); }
  
  auto state = this->game->getState();
  return DoomEnvironment::State(state->screenBuffer);
}

/**
 * Checks if the game episode has reached the terminal state.
 * @param state desired state.
 * @return true if state is a terminal state, otherwise false.
 */
bool DoomEnvironment::IsTerminal(const DoomEnvironment::State& state) const {
  return this->game->isEpisodeFinished() || this->game->isMapEnded() ||
    (this->maxSteps != 0 && this->stepsPerformed >= this->maxSteps);
}

void DoomEnvironment::initGameOptions() {
  // Sets path to vizdoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
  this->game->setViZDoomPath("./bin/vizdoomd.app/Contents/MacOS/vizdoomd");
  // Sets path to doom2 iwad resource file which contains the actual doom game.
  this->game->setDoomGamePath("./bin/doom.wad"); // "../../bin/doom2.wad");
  // Sets path to additional resources iwad file which is basically your scenario iwad.
  // If not specified default doom2 maps will be used and it's pretty much useless... unless you want to play doom.
  this->game->setDoomScenarioPath("./bin/doom.wad");

  // Sets resolution. The original Doom has a resolution of 320x200
  this->game->setScreenResolution(RES_320X200);
  // Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
  this->game->setScreenFormat(RGB24);

  // Other rendering options...
  this->game->setRenderHud(true);
  this->game->setRenderMinimalHud(false); // If hud is enabled
  this->game->setRenderCrosshair(false);
  this->game->setRenderWeapon(true);
  this->game->setRenderDecals(true);
  this->game->setRenderParticles(true);
  this->game->setRenderEffectsSprites(true);
  this->game->setRenderMessages(true);
  this->game->setRenderCorpses(true);
  this->game->setObjectsInfoEnabled(true);
  this->game->setSectorsInfoEnabled(true);

  // Turns on the sound. (turned off by default)
  this->game->setSoundEnabled(true);
  //game->setAudioBufferSize(SR_44100); // Doesn't do anything?

  this->game->setDeathPenalty(10);
  //this->game->setLivingReward(0.01);

  // Makes the window appear (turned on by default)
  this->game->setWindowVisible(true);

  // Enables engine output to console.
  //game->setConsoleEnabled(true);
}

void DoomEnvironment::initGameActions() {
  // Adds buttons that will be allowed.
  // IMPORTANT: Order matters here - this MUST match the ordering of the ActionType enum
  this->game->addAvailableButton(MOVE_LEFT);     // 0
  this->game->addAvailableButton(MOVE_RIGHT);    // 1
  this->game->addAvailableButton(TURN_LEFT);     // 2
  this->game->addAvailableButton(TURN_RIGHT);    // 3
  this->game->addAvailableButton(ATTACK);        // 4
  this->game->addAvailableButton(MOVE_FORWARD);  // 5
  this->game->addAvailableButton(MOVE_BACKWARD); // 6
  this->game->addAvailableButton(USE);           // 7

  // Setup our actionMap that maps the ActionTypes to binary vectors that the ViZDoom game accepts
  this->actionMap.clear();
  // Define our action vectors. Each list entry corresponds to declared buttons
  // game.getAvailableButtonsSize() can be used to check the number of available buttons.
  const auto numActions = game->getAvailableButtonsSize();
  std::vector<std::vector<double>> actions;
  auto actionIdx = 0;
  for (; actionIdx < numActions; actionIdx++) {
    // Columns are the actions defined by addAvailableButton, in order - NOTE: you can use combination actions!
    std::vector<double> actionVec(numActions, 0);
    actionVec[actionIdx] = 1;
    this->actionMap.insert({static_cast<ActionType>(actionIdx), actionVec});
  }
  // Add the combo actions as well (once again, order MUST match the ActionType enum)
  std::vector<double> leftAtk(numActions, 0); leftAtk[0] = 1; leftAtk[4] = 1; 
  this->actionMap.insert({ActionType::DoomActionMoveLeftAndAttack, leftAtk});

  std::vector<double> rightAtk(numActions, 0); rightAtk[1] = 1; rightAtk[4] = 1; 
  this->actionMap.insert({ActionType::DoomActionMoveRightAndAttack, rightAtk});

  std::vector<double> backAtk(numActions, 0); backAtk[6] = 1; backAtk[4] = 1; 
  this->actionMap.insert({ActionType::DoomActionMoveBackwardAndAttack, backAtk});

  std::vector<double> fwdAtk(numActions, 0); fwdAtk[5] = 1; fwdAtk[4] = 1;
  this->actionMap.insert({ActionType::DoomActionMoveForwardAndAttack, fwdAtk});
}

void DoomEnvironment::initGameVariables() {
  // Adds game variables that will be included in state.
  this->game->addAvailableGameVariable(KILLCOUNT);   // Counts the number of monsters killed during the current episode. 
  this->game->addAvailableGameVariable(DAMAGECOUNT); // Counts the damage dealt to monsters/players/bots during the current episode.
  this->game->addAvailableGameVariable(SECRETCOUNT); // Counts the number of secret location/objects discovered during the current episode.
  this->game->addAvailableGameVariable(HEALTH);      // Current player health
  this->game->addAvailableGameVariable(ARMOR);       // Current player armor
  this->game->addAvailableGameVariable(AMMO2);       // Amount of ammo for the pistol

  // Reward functions (how we calculate the reward when specific game variables change)
  auto healthRewardFunc = [](auto oldHealth, auto newHealth) { return newHealth-oldHealth; };
  auto armorRewardFunc = [](auto oldArmor, auto newArmor) {
    return (newArmor > oldArmor ? 1.0 : 0.25) * (newArmor-oldArmor); // Smaller penalty for losing armor vs. health
  };
  auto secretsRewardFunc = [](auto oldNumSecrets, auto newNumSecrets) { return newNumSecrets > oldNumSecrets ? 10 : 0; };
  auto dmgRewardFunc = [](auto oldDmg, auto newDmg) { return newDmg > oldDmg ? 0.1*(newDmg-oldDmg) : 0; };
  auto killCountRewardFunc = [](auto oldKillCount, auto newKillCount) { return newKillCount > oldKillCount ? 10 : 0; };

  // Setup our variable-reward mapping
  this->rewardVars = { 
    { HEALTH,      std::make_shared<DoomRewardVarInt<healthRewardFunc>>(HEALTH, "Player health")       },
    { ARMOR,       std::make_shared<DoomRewardVarInt<armorRewardFunc>>(ARMOR, "Player armor")          },
    { SECRETCOUNT, std::make_shared<DoomRewardVarInt<secretsRewardFunc>>(SECRETCOUNT, "Secrets found") },
    { DAMAGECOUNT, std::make_shared<DoomRewardVarInt<dmgRewardFunc>>(DAMAGECOUNT, "Monster damage")    },
    { KILLCOUNT,   std::make_shared<DoomRewardVarInt<killCountRewardFunc>>(KILLCOUNT, "Kill count")    }
  };
}