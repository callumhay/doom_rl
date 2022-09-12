#ifndef __DOOMRLCMDOPTS_HPP__
#define __DOOMRLCMDOPTS_HPP__

#include <string>
#include <boost/program_options.hpp>

class DoomRLCmdOpts {
public:
  static constexpr char doomMapRandom[] = "random";
  static constexpr char doomMapCycle[] = "cycle";

  size_t numEpisodes;
  size_t stepsExplore;
  size_t stepsSave;
  size_t stepsSync;
  double startEpsilon;
  double epsilonMin;
  double epsilonDecay;
  
  double learningRate;
  double minLearningRate;
  double maxLearningRate;

  bool isExecTesting;

  std::string checkpointFilepath;
  std::string doomMap;

  DoomRLCmdOpts(int argc, char* argv[]);
  void printOpts(std::ostream& stream) const;

  bool isLearningRateConstant() const {
    return this->minLearningRate == this->learningRate && this->maxLearningRate == this->learningRate;
  };

private:
  boost::program_options::options_description desc;
  boost::program_options::variables_map vm;
  void checkOpts();

  template<typename T> 
  void cmdVarCheck(const char* cmd, T& var, T defaultVal, const std::string& desc, T min, T max, bool minInc=true, bool maxInc=true);
};

#endif // __DOOMRLCMDOPTS_HPP__