#include "DoomRLNetVersion.hpp"
#include "DoomEnv.hpp"

std::array<size_t, 3> DoomRLNetVersion::getNetInputSize() const {
  switch (this->majorVerison) {
    case 0: case 3: return {3, 100, 160};
    case 1: case 2: return {3, DoomEnv::State::SCREEN_BUFFER_HEIGHT, DoomEnv::State::SCREEN_BUFFER_WIDTH};
    default: assert(false); break;
  }
}