# https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/
#[[
target_sources(vizdoom_test
  PRIVATE
    vizdoom_test.cpp
    DoomRewardVariable.hpp
    DoomEnvironment.hpp
    DoomEnvironment.cpp
  #PUBLIC
)
]]

target_sources(torch_test
  PRIVATE
    torch_test.cpp
    DoomEnv.hpp
    DoomEnv.cpp
    DoomGuy.hpp
    DoomGuy.cpp
    DoomGuyNet.hpp
    DoomGuyNet.cpp
    DoomRLLogger.hpp
    DoomRLLogger.cpp
)
