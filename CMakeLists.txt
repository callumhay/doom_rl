cmake_minimum_required(VERSION 3.7...3.24)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(DoomRL VERSION 1.0 DESCRIPTION "Reinforcement learning for the creation of an AI that plays Doom." LANGUAGES CXX)

### C++ standard settings
set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_GENERATOR "Ninja")
message("generator is set to ${CMAKE_GENERATOR}")

### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
  message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

### Default build type
set(default_build_type "Debug")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


add_executable(doom_rl "")
add_executable(doom_rl_tests "")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(Boost_USE_STATIC_LIBS OFF) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED COMPONENTS program_options unit_test_framework serialization) 

#message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
#find_package(TorchVision REQUIRED)

add_subdirectory(extern/ViZDoom)
add_subdirectory(app)
add_subdirectory(tests)

#set_target_properties(vizdoom_test PROPERTIES LINK_DEPENDS_NO_SHARED true)
#target_include_directories(vizdoom_test PRIVATE extern/ViZDoom/include ${ARMADILLO_INCLUDE_DIR} ${MLPACK_INCLUDE_DIRS})
#target_link_libraries(vizdoom_test PRIVATE libvizdoom_static ${MLPACK_LIBRARIES})

target_include_directories(doom_rl PRIVATE extern/ViZDoom/include ${Torch_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(doom_rl PRIVATE ${TORCH_LIBRARIES} ${Boost_LIBRARIES} libvizdoom_static)

target_include_directories(doom_rl_tests PRIVATE ${Torch_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(doom_rl_tests PRIVATE ${TORCH_LIBRARIES} ${Boost_LIBRARIES})
