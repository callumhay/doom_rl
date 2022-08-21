# https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/

target_sources(doom_rl
  PRIVATE
    doom_rl.cpp
    DoomEnv.hpp
    DoomEnv.cpp
    DoomGuy.hpp
    DoomGuy.cpp
    DoomGuyNet.hpp
    DoomGuyNet.cpp
    DoomRLLogger.hpp
    DoomRLLogger.cpp
    DoomRLCmdOpts.hpp
    DoomRLCmdOpts.cpp
    RNG.hpp
    RNG.cpp
    CosAnnealLRScheduler.hpp
    CosAnnealLRScheduler.cpp
)
