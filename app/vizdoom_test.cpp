#include "ViZDoom.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>

#include "DoomRewardVariable.hpp"
#include "DoomEnvironment.hpp"

template<typename T> class TD; 

using namespace vizdoom;

int main() {

  std::cout << "STARTING VIZDOOM ENVIRONMENT..." << std::endl;

  // Create DoomGame instance. It will run the game and communicate with you.
  DoomGame *game = new DoomGame();

  // Sets path to vizdoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
  game->setViZDoomPath("./bin/vizdoomd.app/Contents/MacOS/vizdoomd");

  // Sets path to doom2 iwad resource file which contains the actual doom game-> Default is "./doom2.wad".
  game->setDoomGamePath("./bin/doom.wad");
  //game->setDoomGamePath("../../bin/doom2.wad");      // Not provided with environment due to licences.

  // Sets path to additional resources iwad file which is basically your scenario iwad.
  // If not specified default doom2 maps will be used and it's pretty much useless... unless you want to play doom.
  game->setDoomScenarioPath("./bin/doom.wad");

  // Set map to start (scenario .wad files can contain many maps).
  game->setDoomMap("E1M1");

  // Sets resolution. The original Doom has a resolution of 320x200
  game->setScreenResolution(RES_320X200);

  // Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
  game->setScreenFormat(RGB24);

  // Sets other rendering options
  game->setRenderHud(true);
  game->setRenderMinimalHud(false); // If hud is enabled
  game->setRenderCrosshair(false);
  game->setRenderWeapon(true);
  game->setRenderDecals(true);
  game->setRenderParticles(true);
  game->setRenderEffectsSprites(true);
  game->setRenderMessages(true);
  game->setRenderCorpses(true);

  game->setObjectsInfoEnabled(true);
  game->setSectorsInfoEnabled(true);

  game->setDeathPenalty(10);
  //game->setLivingReward(0.01);

  // Adds buttons that will be allowed.
  game->addAvailableButton(MOVE_LEFT);     // 0
  game->addAvailableButton(MOVE_RIGHT);    // 1
  game->addAvailableButton(TURN_LEFT);     // 2
  game->addAvailableButton(TURN_RIGHT);    // 3
  game->addAvailableButton(ATTACK);        // 4
  game->addAvailableButton(MOVE_FORWARD);  // 5
  game->addAvailableButton(MOVE_BACKWARD); // 6
  game->addAvailableButton(USE);           // 7

  // Define some actions. Each list entry corresponds to declared buttons
  // game.getAvailableButtonsSize() can be used to check the number of available buttons.
  // More combinations are naturally possible but only 3 are included for transparency when watching.
  const auto numActions = game->getAvailableButtonsSize();
  std::vector<std::vector<double>> actions;
  for (auto i = 0; i < numActions; i++) {
    // Columns are the actions defined by addAvailableButton, in order - NOTE: you can use combination actions!
    std::vector<double> actionVec(numActions, 0);
    actionVec[i] = 1;
    actions.push_back(actionVec);
  }
  // Add some combo actions...
  // LEFT + ATTACK
  std::vector<double> leftAtk(numActions, 0); leftAtk[0] = 1; leftAtk[4] = 1; 
  actions.push_back(leftAtk);
  // RIGHT + ATTACK
  std::vector<double> rightAtk(numActions, 0); rightAtk[1] = 1; rightAtk[4] = 1; 
  // FORWARD + ATTACK
  std::vector<double> fwdAtk(numActions, 0); fwdAtk[5] = 1; fwdAtk[4] = 1; 
  actions.push_back(fwdAtk);
  actions.push_back(rightAtk);
  // BACKWARD + ATTACK
  std::vector<double> backAtk(numActions, 0); backAtk[6] = 1; backAtk[4] = 1; 
  actions.push_back(backAtk);

  // Adds game variables that will be included in state.
  game->addAvailableGameVariable(KILLCOUNT);   // Counts the number of monsters killed during the current episode. 
  game->addAvailableGameVariable(DAMAGECOUNT); // Counts the damage dealt to monsters/players/bots during the current episode.
  game->addAvailableGameVariable(SECRETCOUNT); // Counts the number of secret location/objects discovered during the current episode.
  game->addAvailableGameVariable(HEALTH);      // Current player health
  game->addAvailableGameVariable(ARMOR);       // Current player armor
  game->addAvailableGameVariable(AMMO2);       // Amount of ammo for the pistol

  // Causes episodes to finish after x tics (actions)
  game->setEpisodeTimeout(3000);

  // Makes episodes start after 10 tics (~after raising the weapon)
  game->setEpisodeStartTime(10);

  // Makes the window appear (turned on by default)
  game->setWindowVisible(true);

  // Turns on the sound. (turned off by default)
  game->setSoundEnabled(true);
  //game->setAudioBufferSize(SR_44100); // Doesn't do anything?

  // Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
  game->setMode(ASYNC_SPECTATOR);

  // Enables engine output to console.
  //game->setConsoleEnabled(true);

  // Initialize the game. Further configuration won't take any effect from now on.
  game->init();

  std::srand(time(0));

  // Run this many episodes
  int episodes = 10;

  // Sets time that will pause the engine after each action.
  // Without this everything would go too fast for you to keep track of what's happening.
  unsigned int sleepTime = 1000 / DEFAULT_TICRATE; // = 28

  // Reward functions (how we calculate the reward when specific game variables change)
  auto healthRewardFunc = [](auto oldHealth, auto newHealth) { return newHealth-oldHealth; };
  auto armorRewardFunc = [](auto oldArmor, auto newArmor) {
    return (newArmor > oldArmor ? 1.0 : 0.25) * (newArmor-oldArmor); // Smaller penalty for losing armor vs. health
  };
  auto secretsRewardFunc = [](auto oldNumSecrets, auto newNumSecrets) { return newNumSecrets > oldNumSecrets ? 10 : 0; };
  auto dmgRewardFunc = [](auto oldDmg, auto newDmg) { return newDmg > oldDmg ? 0.1*(newDmg-oldDmg) : 0; };
  auto killCountRewardFunc = [](auto oldKillCount, auto newKillCount) { return newKillCount > oldKillCount ? 10 : 0; };

  // Map of all our game state variables that are tied to reward calculations, we will iterate through this to
  // calculate a reward in any given state during play
  std::unordered_map<vizdoom::GameVariable, std::shared_ptr<DoomRewardVariable>> rewardVars = { 
    { HEALTH,      std::make_shared<DoomRewardVarInt<healthRewardFunc>>(HEALTH, "Player health")       },
    { ARMOR,       std::make_shared<DoomRewardVarInt<armorRewardFunc>>(ARMOR, "Player armor")          },
    { SECRETCOUNT, std::make_shared<DoomRewardVarInt<secretsRewardFunc>>(SECRETCOUNT, "Secrets found") },
    { DAMAGECOUNT, std::make_shared<DoomRewardVarInt<dmgRewardFunc>>(DAMAGECOUNT, "Monster damage")    },
    { KILLCOUNT,   std::make_shared<DoomRewardVarInt<killCountRewardFunc>>(KILLCOUNT, "Kill count")    }
  };

  for (int i = 0; i < episodes; i++) {
    std::cout << "Starting Episode #" << i + 1 << std::endl;

    // Starts a new episode. It is not needed right after init() but it doesn't cost much and the loop is nicer.
    game->newEpisode();
    // (Re)initialize all our reward variables
    for (auto& [varType, varPtr] : rewardVars) { varPtr->reinit(game); }
    double totalEpisodeReward = 0.0;

    while (!game->isEpisodeFinished()) {

        // Get the state
        auto state = game->getState(); // GameStatePtr is std::shared_ptr<GameState>

        // Which consists of:
        unsigned int n = state->number;
        //const auto& vars  = state->gameVariables;
        //ImageBufferPtr screenBuf  = state->screenBuffer;
        //ImageBufferPtr depthBuf   = state->depthBuffer;
        //ImageBufferPtr labelsBuf  = state->labelsBuffer;
        //ImageBufferPtr automapBuf = state->automapBuffer;

        // BufferPtr is std::shared_ptr<Buffer> where Buffer is std::vector<uint8_t>
        //std::vector<Label> labels = state->labels;

        // Take a random action and get a reward (if any)
        auto currActionReward = game->makeAction(actions[std::rand() % actions.size()]);

        // You can also get last reward by using this function
        // double reward = game->getLastReward();

        // Makes a "prolonged" action and skip frames.
        //int skiprate = 4
        //double reward = game->makeAction(choice(actions), skiprate)

        // Calculate the current action step's reward based on the the change in the game state
        double currStateReward = 0;
        for (auto& [varType, varPtr] : rewardVars) { 
          currStateReward += varPtr->updateAndCalculateReward(game);
        }

        auto mapEnded = game->isMapEnded();
        currStateReward += mapEnded ? 100 : 0;
        
        if ((currStateReward + currActionReward) != 0) {
          std::cout << "State #" << n << std::endl;
          std::cout << "Action reward: " << currActionReward << std::endl;
          std::cout << "Game variable reward: " << currStateReward << std::endl;
          std::cout << "=====================" << std::endl;

          totalEpisodeReward += currStateReward;
        }
        
        if (mapEnded) { break; }
        else if(sleepTime) { std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime)); }
    }

    totalEpisodeReward += game->getTotalReward(); 

    std::cout << "Episode finished.\n";
    std::cout << "Total reward: " << totalEpisodeReward << "\n";
    std::cout << "************************\n";
  }

  // It will be done automatically in destructor but after close You can init it again with different settings.
  game->close();
  delete game;
}