#ifndef __DOOMRLLOGGER_HPP__
#define __DOOMRLLOGGER_HPP__

#include <ctime>
#include <string>
#include <vector>

class DoomRLCmdOpts;

class DoomRLLogger {
public:
  DoomRLLogger(const std::string& logDir, const std::string& checkpointDir);

  void logStartSession(const DoomRLCmdOpts& cmdOpts);
  void logStep(double reward, double loss, double q,  double lr, double epsilon);
  void logEpisode(size_t episodeNum, size_t stepNum);

private:
  bool hasStartedLogging;

  std::string logFilepath;
  std::string csvFilepath;

  std::time_t recordTime;

  size_t currEpLength;
  double currEpAvgLearningRate;
  double currEpAvgQ;
  double currEpAvgLoss;
  size_t currEpLossLength;
  double currEpReward;
  double currEpAvgEpsilon;

  void initEpisode() {
    this->currEpAvgLearningRate = 0.0;
    this->currEpAvgQ = 0.0;
    this->currEpAvgLoss = 0.0;
    this->currEpLossLength = 0;
    this->currEpReward = 0.0;
    this->currEpLength = 0;
    this->currEpAvgEpsilon = 0.0;
  };

  void logPreamble() const;
};

#endif // __DOOMRLLOGGER_HPP__