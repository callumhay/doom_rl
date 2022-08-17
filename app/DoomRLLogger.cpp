#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <algorithm>
#include <numeric>

#include <torch/torch.h>

#include "DoomRLLogger.hpp"

constexpr size_t MAX_RUNNING_AVG_SIZE = 100;
constexpr std::array<int, 9> logWidths = {8, 8, 10, 15, 15, 15, 15, 15, 20};
constexpr size_t totalLogWidth = std::accumulate(logWidths.cbegin(), logWidths.cend(), 0);
/*
template <typename T>
void pushRunningAvgData(std::vector<T>& runningAvg, T data) {
  if (runningAvg.size() == MAX_RUNNING_AVG_SIZE) {
    std::swap(runningAvg, runningAvg.back());
    runningAvg.pop_back();
  }
  runningAvg.push_back(data);
}
*/

void headerToSStream(std::stringstream& ss, const std::string separator="") {
  ss << std::setw(logWidths[0]) << "Episode" << separator 
     << std::setw(logWidths[1]) << "Step" << separator
     << std::setw(logWidths[2]) << "Epsilon" << separator 
     << std::setw(logWidths[3]) << "Mean Reward" << separator
     << std::setw(logWidths[4]) << "Mean Length" << separator
     << std::setw(logWidths[5]) << "Mean Loss" << separator
     << std::setw(logWidths[6]) << "Mean Q-Value" << separator
     << std::setw(logWidths[7]) << "Time Delta" << separator
     << std::setw(logWidths[8]) << "Time" << std::endl;
}
std::string toTimeStr(std::time_t t) {
  std::stringstream timeSS;
  timeSS << std::put_time(std::localtime(&t), "%c");
  return timeSS.str();
}

DoomRLLogger::DoomRLLogger(const std::string& saveDir): 
saveDir(saveDir), saveLogFilepath(saveDir + "/doom_rl_log.txt"), 
recordTime(std::time(nullptr)), hasStartedLogging(false) {

  auto timeStr = toTimeStr(this->recordTime);
  std::replace(timeStr.begin(), timeStr.end(), ' ', '_');
  this->csvFilepath = saveDir + "/doom_rl_log_" + timeStr + ".csv";

  this->initEpisode();
}

void DoomRLLogger::logStep(double reward, double loss, double q) {
  if (!this->hasStartedLogging) {
    this->logPreamble();
    this->hasStartedLogging = true;
  }

  this->currEpReward += reward;
  this->currEpLength++;
  if (loss > 0) {
    this->currEpLoss += loss;
    this->currEpLossLength++;
    this->currEpQ += q;
  }
}

void DoomRLLogger::logEpisode() {
  this->epRewards.push_back(this->currEpReward);
  this->epLengths.push_back(this->currEpLength);
  auto epAvgLoss = 0.0;
  auto epAvgQ = 0.0;
  if (this->currEpLossLength != 0) {
    epAvgLoss = std::round(this->currEpLoss/this->currEpLossLength);
    epAvgQ = std::round(this->currEpQ / this->currEpLossLength);
  }
  this->epAvgLosses.push_back(epAvgLoss);
  this->epAvgQs.push_back(epAvgQ);

  this->initEpisode();
}

void DoomRLLogger::record(size_t episodeNum, size_t stepNum, double epsilon) {
  auto meanEpReward  = torch::from_blob(this->epRewards.data(),   {static_cast<int>(this->epRewards.size())}).mean();
  auto meanEpLength  = torch::from_blob(this->epLengths.data(),   {static_cast<int>(this->epLengths.size())}).mean();
  auto meanEpAvgLoss = torch::from_blob(this->epAvgLosses.data(), {static_cast<int>(this->epAvgLosses.size())}).mean();
  auto meanEpAvgQ    = torch::from_blob(this->epAvgQs.data(),     {static_cast<int>(this->epAvgQs.size())}).mean();

  auto lastRecordTime = this->recordTime;
  this->recordTime = std::time(nullptr);
  auto timeSinceLastRecord = this->recordTime-lastRecordTime;

  std::stringstream headerSS;
  headerToSStream(headerSS);
  std::cout << headerSS.rdbuf();

  std::stringstream recordSS;
  recordSS << std::setw(logWidths[0]) << episodeNum 
           << std::setw(logWidths[1]) << stepNum 
           << std::setw(logWidths[2]) << std::setprecision(3) << epsilon 
           << std::setw(logWidths[3]) << std::setprecision(3) << meanEpReward
           << std::setw(logWidths[4]) << std::setprecision(3) << meanEpLength 
           << std::setw(logWidths[5]) << std::setprecision(3) << meanEpAvgLoss 
           << std::setw(logWidths[6]) << std::setprecision(3) << meanEpAvgQ 
           << std::setw(logWidths[7]) << timeSinceLastRecord 
           << std::setw(logWidths[8]) << std::put_time(std::localtime(&this->recordTime), "%c")
           << std::endl;

  std::ofstream ofs(this->saveLogFilepath.c_str(), std::ios_base::out | std::ios_base::app);
  ofs << recordSS.rdbuf();
  ofs.close();

  this->epRewards.clear();
  this->epLengths.clear();
  this->epAvgLosses.clear();
  this->epAvgQs.clear();
}

void DoomRLLogger::logPreamble() const {
  auto timeStr = toTimeStr(this->recordTime);

  // Write a headers to the log and csv files...
  {
    std::ofstream logOfs(this->saveLogFilepath.c_str(), std::ios_base::out | std::ios_base::app);
    logOfs << std::string(totalLogWidth, '*') << std::endl;
    logOfs << "[" << timeStr << "] Appending new log..." << std::endl;
    logOfs << std::string(totalLogWidth, '-') << std::endl;

    std::stringstream ss;
    headerToSStream(ss);
    logOfs << ss.rdbuf();

    logOfs.close();
  }
  // Start a new csv file
  {
    std::ofstream csvOfs(this->csvFilepath.c_str(), std::ios_base::out | std::ios_base::app);
    std::stringstream ss;
    headerToSStream(ss, ",");
    csvOfs << ss.rdbuf();
    csvOfs.close();
  }
}