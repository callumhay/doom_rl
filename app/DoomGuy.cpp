#include <assert.h>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <regex>

#include "debug_doom_rl.hpp"
#include "RNG.hpp"
#include "DoomGuy.hpp"
#include "ReplayMemory.hpp"


namespace fs = std::filesystem;

using State  = DoomEnv::State;
using Action = DoomEnv::Action;

DoomGuy::DoomGuy(
  const std::string& saveDir, size_t stepsPerEpisode, 
  size_t stepsExplore, size_t stepsBetweenSaves, size_t stepsBetweenSyncs, 
  double startEpsilon, double epsilonMin, double epsilonDecay, double learningRate
) : 
saveDir(saveDir), saveNumber(0), currStep(0), 
stepsPerEpisode(stepsPerEpisode), stepsBetweenLearning(4),
stepsBetweenSaves(stepsBetweenSaves), stepsBetweenSyncs(stepsBetweenSyncs), stepsExplore(stepsExplore),
epsilon(startEpsilon), epsilonDecayMultiplier(epsilonDecay), epsilonMin(epsilonMin),
useCuda(torch::cuda::is_available()), gamma(0.95), lrScheduler(nullptr), 
net(State::NUM_CHANNELS, DoomEnv::getNumActions(DoomGuyNetImpl::version), DoomGuyNetImpl::version),
optimizer(net->parameters(), torch::optim::AdamOptions(learningRate)) {

  //for (auto params : this->net->parameters()) {
    //std::cout << params << std::endl;
  //}

  if (this->useCuda) { 
    this->net->to(torch::kCUDA);
    this->lossFn->to(torch::kCUDA);
  }
  this->rebuildScheduler();
  //this->lrScheduler->setEnabled(false);
}

/**
 * Given a state, choose an epsilon-greedy action and update value of step.
 */
Action DoomGuy::act(torch::Tensor state, std::unique_ptr<DoomEnv>& env) {
  #ifdef DEBUG
  auto [TENSOR_INPUT_HEIGHT, TENSOR_INPUT_WIDTH] = DoomEnv::State::TENSOR_INPUT_HEIGHT_WIDTH(this->net->getCurrVersion());
  assert((state.sizes() == torch::IntArrayRef({DoomEnv::State::NUM_CHANNELS, TENSOR_INPUT_HEIGHT, TENSOR_INPUT_WIDTH})));
  #endif

  Action action;// = Action::DoomNoAction;

  // If the player is guiding the actions then we just look at what last action taken was
  //if (env->isInActivePlayMode()) {
  //  action = env->getLastAction();
  //  env->advanceActionFrames();
  //}
  //else {

  auto random = RNG::getInstance()->randZeroToOne();
  if (random < this->epsilon) {
    // Explore
    action = static_cast<Action>(RNG::getInstance()->rand(0, this->net->getOutputDim()-1));
  }
  else {
    torch::NoGradGuard no_grad;
    
    // Exploit
    auto stateTensor = state.unsqueeze(0);

    #ifdef DEBUG
    assert((stateTensor.sizes() == torch::IntArrayRef({1, DoomEnv::State::NUM_CHANNELS, TENSOR_INPUT_HEIGHT, TENSOR_INPUT_WIDTH})));
    #endif

    auto actionTensor = this->net->forward(this->useCuda ? stateTensor.cuda() : stateTensor, DoomGuyNetImpl::Model::Online);
    auto bestActionTensor = torch::argmax(actionTensor, 1);
    action = static_cast<DoomEnv::Action>(bestActionTensor.item<int>());
  }
  // Decay epsilon
  this->epsilon = std::max(this->epsilonMin, this->epsilon*this->epsilonDecayMultiplier);

  // Increment the step
  this->currStep++;
  this->lrScheduler->step();

  return action;
}

std::tuple<double, double> DoomGuy::learn(std::unique_ptr<ReplayMemory>& replayMemory, size_t batchSize) {
  if (this->currStep % this->stepsBetweenSyncs == 0) {
    std::cout << "Synchronizing the target network with the online network." << std::endl;
    this->net->syncTarget(); // Current online network get loaded into the target network
  }
  if (this->currStep % this->stepsBetweenSaves == 0) {
    this->save(); // Save the agent network to disk
  }
  if (this->currStep < this->stepsExplore || this->currStep % this->stepsBetweenLearning != 0) {
    return std::make_tuple(-1.0,-1.0);
  }

  // Sample from memory
  auto [stateBatch, nextStateBatch, actionBatch, rewardBatch, doneBatch] = replayMemory->randomRecall(batchSize);
  if (this->useCuda) {
    stateBatch = stateBatch.cuda();
    nextStateBatch = nextStateBatch.cuda();
    actionBatch = actionBatch.cuda();
    rewardBatch = rewardBatch.cuda();
    doneBatch = doneBatch.cuda();
  }

  #ifdef DEBUG
  auto [TENSOR_INPUT_HEIGHT, TENSOR_INPUT_WIDTH] = DoomEnv::State::TENSOR_INPUT_HEIGHT_WIDTH(this->net->getCurrVersion());
  assert((stateBatch.sizes() == torch::IntArrayRef({
    static_cast<int>(batchSize), DoomEnv::State::NUM_CHANNELS, TENSOR_INPUT_HEIGHT, TENSOR_INPUT_WIDTH
  })));
  assert((nextStateBatch.sizes() == torch::IntArrayRef({
    static_cast<int>(batchSize), DoomEnv::State::NUM_CHANNELS, TENSOR_INPUT_HEIGHT, TENSOR_INPUT_WIDTH
  })));
  assert((rewardBatch.sizes() == torch::IntArrayRef({static_cast<int>(batchSize), 1})));
  assert((doneBatch.sizes() == torch::IntArrayRef({static_cast<int>(batchSize), 1})));
  #endif

  // Get the Temporal Difference (TD) estimate and target
  auto tdEst = this->tdEstimate(stateBatch, actionBatch);
  assert((tdEst.sizes() == torch::IntArrayRef({static_cast<int>(batchSize), 1})));
  auto tdTgt = this->tdTarget(rewardBatch, nextStateBatch, doneBatch);
  assert((tdTgt.sizes() == torch::IntArrayRef({static_cast<int>(batchSize), 1})));

  // Backprogogate the loss through the Q-Online Network
  auto loss = this->updateQOnline(tdEst, tdTgt);
  auto lossScalar = loss.toDouble();

  this->lrScheduler->onBatchEnd(lossScalar);

  return std::make_tuple(tdEst.mean().item<double>(), lossScalar);
}

void DoomGuy::episodeEnded(size_t episodeNum) {
  if (this->currStep < this->stepsExplore) { return; }
  this->lrScheduler->onEpochEnd(episodeNum);
}


std::string DoomGuy::optimizerSaveFilepath(size_t version, size_t saveNum) {
  std::stringstream optimSavePath;
  optimSavePath << this->saveDir << "/optimizer_v" << version << "_save_" << saveNum << ".chkpt";
  return optimSavePath.str();
}

std::string DoomGuy::netSaveFilepath(size_t version, size_t saveNum) {
  std::stringstream netSavePath;
  netSavePath << this->saveDir << "/network_v" << version << "_save_" << saveNum << ".chkpt";
  return netSavePath.str();
}

void DoomGuy::save() {
  fs::create_directories(this->saveDir); // Make sure the save path exists

  this->saveNumber++;

  auto netSavePath = this->netSaveFilepath(this->net->getCurrVersion(), this->saveNumber);
  this->net->to(torch::kCPU);
  torch::save(this->net, netSavePath); // Make sure the network is always saved with the cpu version (so we can load it elsewhere!)
  if (this->useCuda) { this->net->to(torch::kCUDA); }      // Back to using cuda (if available)

  auto optimSavePath = this->optimizerSaveFilepath(this->net->getCurrVersion(), this->saveNumber);
  torch::save(this->optimizer, optimSavePath);

  std::cout << "[Step " << this->currStep << "]:" << std::endl
            << "\tDoomGuyNet saved to " << netSavePath << std::endl
            << "\tOptimizer saved to  " << optimSavePath << std::endl;
}

void DoomGuy::load(const std::string& chkptFilepath) {

  auto chkptVersion = 0; // Default version for the original checkpoint files is 0
  // See what version the checkpoint file is...
  std::regex vRe("_v([[:digit:]]+)[_\\.]");
  std::smatch vMatches;
  if (std::regex_search(chkptFilepath, vMatches, vRe)) {
    chkptVersion = std::stoi(vMatches[1].str());
    if (chkptVersion < 0 || chkptVersion > DoomGuyNetImpl::version) {
      std::cerr << "Invalid checkpoint file version found: " << chkptVersion << ". "
                << "Current support is from versions 0 to " << DoomGuyNetImpl::version << "." << std::endl;
      std::cerr << "Ignoring checkpoint file " << chkptFilepath << std::endl;
      return;
    }
  }

  if (chkptVersion != this->net->getCurrVersion()) {
    // Network needs to be rebuilt under a different version
    this->net = DoomGuyNet(State::NUM_CHANNELS, DoomEnv::getNumActions(chkptVersion), chkptVersion);
  }

  torch::load(this->net, chkptFilepath);
  std::cout << "Loaded checkpoint from file " << chkptFilepath << "!" << std::endl;

  // Check to see if there's an optimizer file to load as well
  std::regex numRe("(.*)network_v[[:digit:]]+_save_([[:digit:]]+)\\.");
  std::smatch numMatches;
  if (std::regex_search(chkptFilepath, numMatches, numRe)) {
    auto path = numMatches[1].str();
    auto numStr = numMatches[2].str();

    std::stringstream optimFilepathSS;
    optimFilepathSS << path << "optimizer_v" << this->net->getCurrVersion() << "_save_" << numStr << ".chkpt";
    auto optimFilepath = optimFilepathSS.str();

    if (std::filesystem::exists(optimFilepath)) {
      torch::load(this->optimizer, optimFilepath);
      std::cout << "Loaded optimizer from file " << optimFilepath << "!" << std::endl;
      this->rebuildScheduler();
    }
    else {
      std::cout << "Warning: No optimizer found at expected file path (" << optimFilepath << ")." << std::endl;
      this->optimizer.defaults().set_lr(this->lrScheduler->calcLR());
    }
  }
  
  this->net->freezeTarget();
  if (this->useCuda) { this->net->to(torch::kCUDA); }
}

torch::Tensor DoomGuy::tdEstimate(torch::Tensor stateBatch, torch::Tensor actionBatch) {

  // Get the network output, a tensor of the form [batch_size, Q_values]
  // Where Q_Values are the online network Q-Values for each action, with a dimension of DoomEnv::numActions
  auto currentQ = this->net->forward(stateBatch, DoomGuyNetImpl::Model::Online);
  assert((currentQ.sizes() == torch::IntArrayRef({stateBatch.sizes()[0], static_cast<int>(this->net->getOutputDim())})));

  // We need to select the Q-values in currentQ for the given batch of actions i.e., index into each batch
  // by the actions specified in actionBatch
  auto gatheredQ = currentQ.gather(1, actionBatch);
  assert((gatheredQ.sizes() == torch::IntArrayRef({actionBatch.sizes()[0], 1})));

  return gatheredQ;
}

torch::Tensor DoomGuy::tdTarget(torch::Tensor rewardBatch, torch::Tensor nextStateBatch, torch::Tensor doneBatch) {
  torch::NoGradGuard no_grad;
  
  auto nextQOnline = this->net->forward(nextStateBatch, DoomGuyNetImpl::Model::Online);
  auto nextQTarget = this->net->forward(nextStateBatch, DoomGuyNetImpl::Model::Target);
  assert((nextQOnline.sizes() == torch::IntArrayRef({nextStateBatch.sizes()[0], static_cast<int>(this->net->getOutputDim())})));
  assert((nextQTarget.sizes() == torch::IntArrayRef({nextStateBatch.sizes()[0], static_cast<int>(this->net->getOutputDim())})));

  auto nextQOnlineBestActionIndices = std::get<1>(nextQOnline.max(1)).unsqueeze(1);
  auto bestNextQ = nextQTarget.gather(1, nextQOnlineBestActionIndices);
  assert((bestNextQ.sizes() == torch::IntArrayRef({nextQTarget.sizes()[0], 1})));

  return rewardBatch + this->gamma * bestNextQ * (1 - doneBatch);
}

torch::Scalar DoomGuy::updateQOnline(torch::Tensor tdEstimate, torch::Tensor tdTarget) {
  // NOTE: Both tdEstimate and tdTarget are tensors of size [DEFAULT_REPLAY_BATCH_SIZE, 1]
  auto loss = this->lossFn(tdEstimate, tdTarget.detach());
  
  std::vector<torch::Tensor> befores;
  std::vector<std::string> layerNames;
  auto layerCount = 0;
  for (const auto& params : this->net->online->parameters()) {
    befores.push_back(params.clone());
    layerCount++;
    //layerNames.push_back(this->net->online[layerCount++]->name());
  }

  //std::cout << this->net->online->parameters().size() << std::endl;
  //auto beforeHasGrad = this->net->parameters()[0].grad();
  this->optimizer.zero_grad();
  loss.backward();
  this->optimizer.step();

  std::vector<torch::Tensor> afters;
  for (auto i = 0; i < befores.size(); i++) {
    afters.push_back(this->net->online->parameters()[i].clone());
  }
  //auto afterHasGrad = this->net->parameters()[0].grad();

  //std::cout << before << std::endl;
  //std::cout << after << std::endl;

  if (this->currStep % 500 == 0) {
    std::vector<std::string> equalLayers;
    for (auto i = 0; i < befores.size(); i++) {
      if (torch::equal(befores[i], afters[i])) {
        equalLayers.push_back(std::to_string(i));// + ": " + layerNames[i]);
      }
    }
    std::stringstream ss;
    ss << "Warning: Online network isn't changing for " << equalLayers.size() << "/" << layerCount << " layers: ";
    std::copy(equalLayers.begin(), equalLayers.end(), std::ostream_iterator<std::string>(ss, ", "));
    //ss << " (possible layers: [";
    //std::copy(checkLayerIndices.begin(), checkLayerIndices.end(), std::ostream_iterator<int>(ss, ","));
    //ss << "])";
    std::cout << "[Step: " << this->currStep << "]: " 
              << (equalLayers.size() != 0 ? ss.str() : "Online network is changing on all layers.") << std::endl
              << "Current learning rate is " << std::fixed << std::setprecision(6) << this->getLearningRate() << std::endl
              << "Current loss is " << loss.item<double>() << std::endl;
  }

  return loss.item();
}
