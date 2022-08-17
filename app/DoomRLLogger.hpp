#ifndef __DOOMRLLOGGER_HPP__
#define __DOOMRLLOGGER_HPP__

#include <ctime>
#include <string>
#include <vector>

class DoomRLLogger {
public:
  DoomRLLogger(const std::string& saveDir);

  void logStep(double reward, double loss, double q);
  void logEpisode();

  void record(size_t episodeNum, size_t stepNum, double epsilon);

private:
  bool hasStartedLogging;

  std::string saveDir;
  std::string saveLogFilepath;
  std::string csvFilepath;

  std::time_t recordTime;

  double currEpReward;
  size_t currEpLength;
  double currEpLoss;
  double currEpQ;
  size_t currEpLossLength;

  std::vector<double> epRewards;
  std::vector<size_t> epLengths;
  std::vector<double> epAvgLosses;
  std::vector<double> epAvgQs;

  void initEpisode() {
    this->currEpReward = 0.0;
    this->currEpLength = 0;
    this->currEpLoss = 0.0;
    this->currEpQ = 0.0;
    this->currEpLossLength = 0;
  };

  void logPreamble() const;


};

#endif // __DOOMRLLOGGER_HPP__