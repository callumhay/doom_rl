#include <fstream>
#include <sstream>
#include <filesystem>
#include <array>
#include <algorithm>
#include <numeric>

#include <torch/torch.h>

#include "DoomRLLogger.hpp"
#include "DoomRLCmdOpts.hpp"
#include "DoomGuy.hpp"

namespace fs = std::filesystem;

constexpr size_t MAX_RUNNING_AVG_SIZE = 100;
constexpr std::array<size_t, 12> logWidths = {8, 8, 6, 10, 15, 12, 14, 16, 10, 10, 12, 27};

template<typename T, std::size_t N>
constexpr T arraySum(const std::array<T,N>& array) {
  T sum = 0;
  for (std::size_t i = 0; i < N; i++) { sum += array[i]; }
  return sum;
};
constexpr size_t totalLogWidth = arraySum(logWidths);

void headerToSStream(std::stringstream& ss, const std::string separator="") {
  int i = 0;
  ss << std::setw(logWidths[i++]) << "Episode" << separator;
  ss << std::setw(logWidths[i++]) << "Step" << separator;
  ss << std::setw(logWidths[i++]) << "Map" << separator;
  ss << std::setw(logWidths[i++]) << "Epsilon" << separator; 
  ss << std::setw(logWidths[i++]) << "Learning Rate" << separator;
  ss << std::setw(logWidths[i++]) << "Ep Reward" << separator;
  ss << std::setw(logWidths[i++]) << "Total Reward" << separator;
  ss << std::setw(logWidths[i++]) << "Length (Steps)" << separator;
  ss << std::setw(logWidths[i++]) << "Loss" << separator;
  ss << std::setw(logWidths[i++]) << "Q-Value" << separator;
  ss << std::setw(logWidths[i++]) << "Time Delta" << separator;
  ss << std::setw(logWidths[i++]) << "Time" << std::endl;
}
std::string toTimeStr(std::time_t t) {
  std::stringstream timeSS;
  timeSS << std::put_time(std::localtime(&t), "%c");
  return timeSS.str();
}

DoomRLLogger::DoomRLLogger(const std::string& logDir, const std::string& checkpointDir): 
logFilepath(logDir + "/doom_rl_log.txt"), 
recordTime(std::time(nullptr)), hasStartedLogging(false), cumulativeReward(0) {

  fs::create_directories(logDir);
  fs::create_directories(checkpointDir);

  auto timeStr = toTimeStr(this->recordTime);
  std::replace(timeStr.begin(), timeStr.end(), ' ', '_');
  this->csvFilepath = checkpointDir + "/doom_rl_" + timeStr + ".csv";

  this->initEpisode();
}

void DoomRLLogger::logStartSession(const DoomRLCmdOpts& cmdOpts, const DoomGuy& guy) {
  auto timeStr = toTimeStr(this->recordTime);
  std::ofstream logOfs(this->logFilepath.c_str(), std::ios_base::out | std::ios_base::app);
  logOfs << std::string(totalLogWidth, '-') << std::endl;
  logOfs << "[" << timeStr << "] Appending new log..." << std::endl;
  cmdOpts.printOpts(logOfs);
  logOfs << "Network version: " << guy.getNetworkVersion() << std::endl;
  logOfs.close();
}

void DoomRLLogger::logStep(double reward, double loss, double q, double lr, double epsilon) {
  if (!this->hasStartedLogging) {
    this->logPreamble();
    this->hasStartedLogging = true;
  }

  auto cumulativeAvg = [](double x, double prevAvg, auto n) {
    return prevAvg + (x-prevAvg) / static_cast<double>(n+1);
  };

  // Calculate the cumulative average for the learning rate in the current episode 
  this->currEpAvgLearningRate = cumulativeAvg(lr, this->currEpAvgLearningRate, this->currEpLength);
  this->currEpAvgEpsilon = cumulativeAvg(epsilon, this->currEpAvgEpsilon, this->currEpLength);

  if (loss > 0) {
    this->currEpAvgLoss = cumulativeAvg(loss, this->currEpAvgLoss, this->currEpLossLength);
    this->currEpAvgQ = cumulativeAvg(q, this->currEpAvgQ, this->currEpLossLength);
    this->currEpLossLength++; // Make sure this is incremented AFTER calculating cumulative averages (e.g., loss)
  }

  this->currEpReward += reward;
  this->cumulativeReward += reward;
  this->currEpLength++; // Make sure this is incremented AFTER calculating cumulative averages (e.g., learning rate)
}

void DoomRLLogger::logEpisode(size_t episodeNum, size_t stepNum, const std::string& mapName) {
  auto lastRecordTime = this->recordTime;
  this->recordTime = std::time(nullptr);
  auto timeSinceLastRecord = this->recordTime-lastRecordTime;
  auto timeStr = toTimeStr(this->recordTime);

  auto recordToSStream = [=](std::stringstream& ss, const std::string separator="") {
    int i = 0;
    auto epAvgLossIsBig = std::abs(this->currEpAvgLoss) > 100;
    auto epAvgQIsBig    = std::abs(this->currEpAvgQ) > 100;
    auto cumlRwdIsBig   = std::abs(this->cumulativeReward) > 100000;
    
    ss << std::fixed << std::setw(logWidths[i++]) << episodeNum << separator;
    ss << std::fixed << std::setw(logWidths[i++]) << stepNum << separator;
    ss << std::fixed << std::setw(logWidths[i++]) << mapName << separator;
    ss << std::fixed << std::setw(logWidths[i++]) << std::setprecision(4) << this->currEpAvgEpsilon << separator;
    ss << std::fixed << std::setw(logWidths[i++]) << std::setprecision(5) << this->currEpAvgLearningRate << separator;
    ss << std::fixed << std::setw(logWidths[i++]) << std::setprecision(2) << this->currEpReward << separator;
    ss << (cumlRwdIsBig   ? std::scientific : std::fixed) << std::setw(logWidths[i++]) << std::setprecision(cumlRwdIsBig ? 3 : 1) << this->cumulativeReward << separator;
    ss << std::fixed << std::setw(logWidths[i++]) << std::setprecision(0) << static_cast<size_t>(this->currEpLength) << separator;
    ss << (epAvgLossIsBig ? std::scientific : std::fixed) << std::setw(logWidths[i++]) << std::setprecision(epAvgLossIsBig ? 2 : 5) << this->currEpAvgLoss << separator;
    ss << (epAvgQIsBig    ? std::scientific : std::fixed) << std::setw(logWidths[i++]) << std::setprecision(epAvgQIsBig ? 2 : 5)    << this->currEpAvgQ << separator;
    ss << std::fixed << std::setw(logWidths[i++]) << timeSinceLastRecord << separator;
    ss << std::fixed << std::setw(logWidths[i++]) << timeStr;
    ss << std::endl;
  };

  std::ofstream ofs(this->logFilepath.c_str(), std::ios_base::out | std::ios_base::app);
  std::stringstream logSS;
  recordToSStream(logSS);
  ofs << logSS.str();
  ofs.close();

  // Output to console as well
  std::stringstream headerSS;
  headerToSStream(headerSS);
  std::cout << headerSS.str() << logSS.str();

  // Output csv also
  std::ofstream csvOfs(this->csvFilepath.c_str(), std::ios_base::out | std::ios_base::app);
  std::stringstream csvSS;
  recordToSStream(csvSS, ",");
  csvOfs << csvSS.str();
  csvOfs.close();

  this->initEpisode();
}

void DoomRLLogger::logPreamble() const {
  // Write a headers to the log and csv files...
  {
    std::ofstream logOfs(this->logFilepath.c_str(), std::ios_base::out | std::ios_base::app);
    logOfs << std::string(totalLogWidth, '-') << std::endl;
    std::stringstream ss;
    headerToSStream(ss);
    logOfs << ss.str();
    logOfs.close();
  }
  // Start a new csv file
  {
    std::ofstream csvOfs(this->csvFilepath.c_str(), std::ios_base::out | std::ios_base::app);
    std::stringstream ss;
    headerToSStream(ss, ",");
    csvOfs << ss.str();
    csvOfs.close();
  }
}