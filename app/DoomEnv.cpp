#include <iostream>
#include <regex>

#include "RNG.hpp"
#include "DoomEnv.hpp"

using namespace vizdoom;

DoomEnv::State::State(vizdoom::ImageBufferPtr screenBuf) {
  assert(screenBuf != nullptr);
  // Preprocess the game's framebuffer (stored as a flat array of uint8_t)...

  // Convert to a 3D tensor (height x width x channels) then 
  auto tempTensor = torch::from_blob(
    screenBuf->data(), 
    {SCREEN_BUFFER_HEIGHT, SCREEN_BUFFER_WIDTH, NUM_CHANNELS}, 
    torch::TensorOptions().dtype(torch::kUInt8)
  );

  // Reformat the tensor to be in the form that the interpolate function expects: Batch x Channel x Height x Width
  auto preppedTensor = tempTensor.permute({2,0,1}).toType(torch::kFloat).div_(255.0).unsqueeze_(0);
  // sizes (shape) is now [1,channels,height,width], also converted channel values from [0,255] -> [0.0,1.0]
  //std::cout << preppedTensor.sizes() << std::endl;
  auto [TENSOR_INPUT_HEIGHT, TENSOR_INPUT_WIDTH] = TENSOR_INPUT_HEIGHT_WIDTH();
  switch (DoomGuyNet::version) {
    case 0:
      // Downsample the screen buffer tensor - this will make a new tensor (so we don't need to clone anything)
      // NOTE: Only dtypes for Float32, Float64 are supported by interpolate!
      this->screenTensor = torch::nn::functional::interpolate(
        preppedTensor, 
        torch::nn::functional::InterpolateFuncOptions()
        .size(std::vector<int64_t>({TENSOR_INPUT_HEIGHT, TENSOR_INPUT_WIDTH}))
        .align_corners(true)
        .mode(torch::kBilinear)
      );
      break;
    case 1:
      // No downsampling, just maintain the same size as the screen buffer.
      this->screenTensor = preppedTensor;
      break;
    default:
      assert(false);
      break;
  }


  //std::cout << this->screenTensor << std::endl;

  // this->screenTensor is now ready to be fed into the NN, with
  // shape/sizes = [1, NUM_CHANNELS, TENSOR_INPUT_HEIGHT, TENSOR_INPUT_WIDTH]
  assert((this->screenTensor.sizes() == torch::IntArrayRef({1, NUM_CHANNELS, TENSOR_INPUT_HEIGHT, TENSOR_INPUT_WIDTH})));
};

DoomEnv::DoomEnv(size_t maxSteps, size_t frameSkip, const std::string& mapName): 
game(std::make_unique<DoomGame>()), lastState(nullptr), frameSkip(frameSkip), maxSteps(maxSteps), 
stepsPerformed(0), doomMapToLoadNext(mapName) {
  // Setup the ViZDoom game environment...
  this->initGameOptions();
  this->initGameActions();
  this->initGameVariables();

  // Special options, specific to the training scenario:
  // ----------------------------------------------------
  // Set map to start (scenario .wad files can contain many maps).
  this->game->setDoomMap(this->doomMapToLoadNext);
  // Causes episodes to finish after the given number of ticks
  // Just set this to a large value - we want the agent to learn to finish a map in an episode and not be terminated prematurely
  // Make sure that this is strictly less than UINT_MAX (0xffffffff = 4,294,967,295) if not the game will crash/do undefined things.
  this->game->setEpisodeTimeout(999999999); 
  // Makes episodes start after 10 tics (~after raising the weapon)
  this->game->setEpisodeStartTime(10);
  // Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
  this->game->setMode(PLAYER);

  // Disable some stuff... causes ViZDoom to crash on some levels (boo!)
  this->game->setSectorsInfoEnabled(false);

  // Initialize the game. Further configuration won't take any effect from now on.
  this->game->init();
}

constexpr size_t doomEpStartNum  = 1;
constexpr size_t doomEpEndNum    = 4;
constexpr size_t doomMapStartNum = 1;
constexpr size_t doomMapEndNum   = 8;

void DoomEnv::setCycledMap() {
  // Maps in the original Doom are defined by the regex E[1-4]M[1-8], we cycle in-order through the maps
  std::regex mapRegEx("E([[:digit:]]+)M([[:digit:]]+)");
  std::smatch matches;
  if (std::regex_search(this->doomMapToLoadNext, matches, mapRegEx)) {
    auto doomEpNum  = std::stoi(matches[1].str());
    auto doomMapNum = std::stoi(matches[2].str());

    doomMapNum++;
    if (doomMapNum > doomMapEndNum) {
      doomMapNum = doomMapStartNum;
      doomEpNum++;
      if (doomEpNum > doomEpEndNum) {
        doomEpNum = doomEpStartNum;
      }
    }
    assert(doomMapNum >= doomMapStartNum && doomMapNum <= doomMapEndNum);
    assert(doomEpNum >= doomEpStartNum && doomEpNum <= doomEpEndNum);

    std::stringstream mapSS;
    mapSS << "E" << doomEpNum << "M" << doomMapNum;
    this->setMap(mapSS.str());
  }
  else {
    std::cerr << "Current map regex doesn't capture the set map: Cannot cycle through maps!" << std::endl;
  }
}

void DoomEnv::setRandomMap() {
  // Generate a random map from inside DOOM.WAD, options are defined by the following regex: E[1-4]M[1-8]
  

  auto randDoomEp  = RNG::getInstance()->rand(doomEpStartNum, doomEpEndNum);
  auto randDoomMap = RNG::getInstance()->rand(doomMapStartNum, doomMapEndNum);
  std::stringstream mapSS;
  mapSS << "E" << randDoomEp << "M" << randDoomMap;
  this->setMap(mapSS.str());
}

bool DoomEnv::isEpisodeFinished() const {
  auto isTerminalState = (
    (this->maxSteps != 0 && this->stepsPerformed >= this->maxSteps) ||
    this->game->getState() == nullptr || this->game->isEpisodeFinished() || 
    this->game->isPlayerDead() || this->game->isMapEnded()
  );

  if (isTerminalState) {
    std::cout << "Episode finished: ";
    if (this->game->isMapEnded()) {
      std::cout << "Agent completed the map!" << std::endl;
    }
    else if (this->game->isPlayerDead()) {
      std::cout << "Agent died!" << std::endl;
    }
    else {
      std::cout << "Max steps reached in the episode." << std::endl;
    }
  }
  return isTerminalState;
}


void DoomEnv::initGameOptions() {
  // Sets path to vizdoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
  this->game->setViZDoomPath(
    #ifdef __APPLE__
    "./bin/vizdoomd.app/Contents/MacOS/vizdoomd"
    #else
    "./bin/vizdoomd"
    #endif
  );
  // Sets path to doom2 iwad resource file which contains the actual doom game.
  this->game->setDoomGamePath("./bin/doom.wad"); // "../../bin/doom2.wad");
  // Sets path to additional resources iwad file which is basically your scenario iwad.
  // If not specified default doom2 maps will be used and it's pretty much useless... unless you want to play doom.
  this->game->setDoomScenarioPath("./bin/doom.wad");

  // Sets resolution. The original Doom has a resolution of 320x200
  // NOTE: This function was custom added to ViZDoom
  
  this->game->setScreenResolution(State::SCREEN_BUFFER_WIDTH, State::SCREEN_BUFFER_HEIGHT);
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
  //this->game->setSoundEnabled(true);
  
  this->game->setDeathPenalty(50);
  //this->game->setLivingReward(0.01);

  // Makes the window appear (turned on by default)
  this->game->setWindowVisible(true);

  // Enables engine output to console.
  //game->setConsoleEnabled(true);
}

void DoomEnv::initGameActions() {
  // Adds buttons that will be allowed.
  // IMPORTANT: Order matters here - this MUST match the ordering of the Action enum
  this->game->addAvailableButton(MOVE_LEFT);     // 0
  this->game->addAvailableButton(MOVE_RIGHT);    // 1
  this->game->addAvailableButton(TURN_LEFT);     // 2
  this->game->addAvailableButton(TURN_RIGHT);    // 3
  this->game->addAvailableButton(ATTACK);        // 4
  this->game->addAvailableButton(MOVE_FORWARD);  // 5
  this->game->addAvailableButton(MOVE_BACKWARD); // 6
  this->game->addAvailableButton(USE);           // 7

  // Setup our actionMap that maps the Actions to binary vectors that the ViZDoom game accepts
  this->actionMap.clear();
  // Define our action vectors. Each list entry corresponds to declared buttons
  // game.getAvailableButtonsSize() can be used to check the number of available buttons.
  const auto numActions = game->getAvailableButtonsSize();
  std::vector<std::vector<double>> actions;
  auto actionIdx = 0;
  for (; actionIdx < numActions; actionIdx++) {
    // Columns are the actions defined by addAvailableButton, in the order added
    std::vector<double> actionVec(numActions, 0);
    actionVec[actionIdx] = 1;
    this->actionMap.insert({static_cast<Action>(actionIdx), actionVec});
  }
  // Add the combo actions as well (once again, order MUST match the Action enum)
  std::vector<double> leftAtk(numActions, 0); leftAtk[0] = 1; leftAtk[4] = 1; 
  this->actionMap.insert({Action::DoomActionMoveLeftAndAttack, leftAtk});

  std::vector<double> rightAtk(numActions, 0); rightAtk[1] = 1; rightAtk[4] = 1; 
  this->actionMap.insert({Action::DoomActionMoveRightAndAttack, rightAtk});

  std::vector<double> backAtk(numActions, 0); backAtk[6] = 1; backAtk[4] = 1; 
  this->actionMap.insert({Action::DoomActionMoveBackwardAndAttack, backAtk});

  std::vector<double> fwdAtk(numActions, 0); fwdAtk[5] = 1; fwdAtk[4] = 1;
  this->actionMap.insert({Action::DoomActionMoveForwardAndAttack, fwdAtk});
}

void DoomEnv::initGameVariables() {
  // Adds game variables that will be included in state.
  this->game->addAvailableGameVariable(KILLCOUNT);   // Counts the number of monsters killed during the current episode. 
  this->game->addAvailableGameVariable(DAMAGECOUNT); // Counts the damage dealt to monsters/players/bots during the current episode.
  this->game->addAvailableGameVariable(SECRETCOUNT); // Counts the number of secret location/objects discovered during the current episode.
  this->game->addAvailableGameVariable(HEALTH);      // Current player health
  this->game->addAvailableGameVariable(ARMOR);       // Current player armor
  this->game->addAvailableGameVariable(AMMO2);       // Amount of ammo for the pistol

  // Reward functions (how we calculate the reward when specific game variables change)
  auto healthRewardFunc = [](int oldHealth, int newHealth) { return newHealth-oldHealth; };
  auto armorRewardFunc = [](int oldArmor, int newArmor) {
    return (newArmor > oldArmor ? 1.0 : 0.25) * (newArmor-oldArmor); // Smaller penalty for losing armor vs. health
  };
  auto itemRewardFunc = [](int oldItemCount, int newItemCount) { return newItemCount > oldItemCount ? (newItemCount-oldItemCount) : 0; };
  auto secretsRewardFunc = [](int oldNumSecrets, int newNumSecrets) { return newNumSecrets > oldNumSecrets ? 10 : 0; };
  auto dmgRewardFunc = [](int oldDmg, int newDmg) { return newDmg > oldDmg ? 0.1*(newDmg-oldDmg) : 0; };
  auto killCountRewardFunc = [](int oldKillCount, int newKillCount) { return newKillCount > oldKillCount ? 10 : 0; };
  auto ammoRewardFunc = [](int oldAmmo, int newAmmo) { return (newAmmo > oldAmmo ? 1.0 : 0.5) * (newAmmo-oldAmmo); };

  // Setup our variable-reward mapping
  this->rewardVars = std::vector<std::shared_ptr<DoomRewardVariable>>({ 
    std::make_shared<DoomRewardVarInt>(HEALTH, "Player health", healthRewardFunc),
    std::make_shared<DoomRewardVarInt>(ARMOR, "Player armor", armorRewardFunc),
    std::make_shared<DoomRewardVarInt>(ITEMCOUNT, "Item count", itemRewardFunc),
    std::make_shared<DoomRewardVarInt>(SECRETCOUNT, "Secrets found", secretsRewardFunc),
    std::make_shared<DoomRewardVarInt>(DAMAGECOUNT, "Monster/Env damage", dmgRewardFunc),
    std::make_shared<DoomRewardVarInt>(KILLCOUNT, "Kill count", killCountRewardFunc),
    std::make_shared<DoomRewardVarInt>(AMMO2, "Pistol ammo count", ammoRewardFunc),
    std::make_shared<DoomPosRewardVariable>(POSITION_X, POSITION_Y, POSITION_Z) 
  });
}