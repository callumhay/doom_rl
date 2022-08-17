#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <algorithm>
#include <numeric>

#include <torch/torch.h>

#include "DoomRLLogger.hpp"

constexpr size_t MAX_RUNNING_AVG_SIZE = 100;
constexpr std::array<int, 9> logWidths = {8, 8, 10, 15, 15, 15, 15, 15, 40};
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
    epAvgLoss = this->currEpLoss / static_cast<double>(this->currEpLossLength);
    epAvgQ = this->currEpQ / static_cast<double>(this->currEpLossLength);
  }
  this->epAvgLosses.push_back(epAvgLoss);
  this->epAvgQs.push_back(epAvgQ);

  this->initEpisode();
}

double calcMean(std::vector<double> dataVec) {
  auto tensor = torch::from_blob(
    dataVec.data(), 
    {static_cast<int>(dataVec.size())}, 
    torch::kFloat64
  );
  auto mean = torch::mean(tensor);
  return mean.item<double>();
}

void DoomRLLogger::record(size_t episodeNum, size_t stepNum, double epsilon) {
  auto meanEpReward  = calcMean(this->epRewards);
  auto meanEpLength  = calcMean(this->epLengths);
  auto meanEpAvgLoss = calcMean(this->epAvgLosses);
  auto meanEpAvgQ    = calcMean(this->epAvgQs);

  auto lastRecordTime = this->recordTime;
  this->recordTime = std::time(nullptr);
  auto timeSinceLastRecord = this->recordTime-lastRecordTime;
  auto timeStr = toTimeStr(this->recordTime);

  auto recordToSStream = [=](std::stringstream& ss, const std::string separator="") {
    ss << std::setw(logWidths[0]) << episodeNum << separator
      << std::setw(logWidths[1]) << stepNum << separator
      << std::setw(logWidths[2]) << std::setprecision(3) << epsilon << separator
      << std::setw(logWidths[3]) << std::setprecision(3) << meanEpReward << separator
      << std::setw(logWidths[4]) << std::setprecision(0) << static_cast<size_t>(meanEpLength) << separator
      << std::setw(logWidths[5]) << std::setprecision(3) << meanEpAvgLoss << separator
      << std::setw(logWidths[6]) << std::setprecision(3) << meanEpAvgQ << separator
      << std::setw(logWidths[7]) << timeSinceLastRecord << separator
      << std::setw(logWidths[8]) << timeStr
      << std::endl;
  };

  std::ofstream ofs(this->saveLogFilepath.c_str(), std::ios_base::out | std::ios_base::app);
  std::stringstream logSS;
  recordToSStream(logSS);
  ofs << logSS.rdbuf();
  ofs.close();

  // Output to console as well
  std::stringstream headerSS;
  headerToSStream(headerSS);
  std::cout << headerSS.rdbuf() << logSS.rdbuf();

  // Output csv also
  std::ofstream csvOfs(this->csvFilepath.c_str(), std::ios_base::out | std::ios_base::app);
  std::stringstream csvSS;
  recordToSStream(csvSS, ",");
  csvOfs << csvSS.rdbuf();
  csvOfs.close();
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