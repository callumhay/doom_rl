#ifndef __DOOMRLCMDOPTS_HPP__
#define __DOOMRLCMDOPTS_HPP__

#include <string>
#include <boost/program_options.hpp>

class DoomRLCmdOpts {
public:
  size_t numEpisodes;
  size_t stepsPerEpMax;
  size_t stepsExplore;
  size_t stepsSave;
  size_t stepsSync;
  double startEpsilon;
  double epsilonDecay;
  double learningRate;
  std::string checkpointFilepath;

  DoomRLCmdOpts(int argc, char* argv[]);
  void printOpts(std::ostream& stream) const;

private:
  boost::program_options::options_description desc;
  boost::program_options::variables_map vm;
  void checkOpts();

  template<typename T> 
  void cmdVarCheck(const char* cmd, T& var, T defaultVal, const std::string& desc, T min, T max, bool minInc=true, bool maxInc=true);
};

#endif // __DOOMRLCMDOPTS_HPP__