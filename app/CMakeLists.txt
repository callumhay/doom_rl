# https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/
target_sources(test
  PRIVATE
    test.cpp
  #PUBLIC
)

target_sources(vizdoom_test
  PRIVATE
    vizdoom_test.cpp
    DoomRewardVariable.hpp
    DoomEnvironment.hpp
    DoomEnvironment.cpp
  #PUBLIC
)

target_sources(mlpack_test
  PRIVATE
    mlpack_test.cpp
)