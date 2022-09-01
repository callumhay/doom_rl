# https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/

add_subdirectory(lr_schedulers)
add_subdirectory(modules)
add_subdirectory(utils)

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
    ReplayMemory.hpp
    debug_doom_rl.hpp
)
