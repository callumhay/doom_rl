#ifndef __DOOMRLVERSION_HPP__
#define __DOOMRLVERSION_HPP__

#include "utils/SingletonT.hpp"

class DoomRLNetVersion : public SingletonT<DoomRLNetVersion> {
public:
  DoomRLNetVersion() {}

  size_t getMajorVersion() const { return this->majorVersion; }
  size_t getMinorVersion() const { return this->minorVerison; }

  // Get the [channels,height,width] of screen buffer that will be accepted by the current version of the network
  std::array<size_t, 3> getNetInputSize() const;




private:
  size_t majorVersion, minorVersion;
  

};


#endif // __DOOMRLVERSION_HPP__
