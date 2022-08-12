cmake_minimum_required(VERSION 3.7...3.24)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(DoomRL VERSION 1.0 DESCRIPTION "Reinforcement learning for the creation of an AI that plays Doom." LANGUAGES CXX)

### C++ standard settings
set(CMAKE_CXX_STANDARD 20 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

add_executable(test "") # NOTE: Sources/headers are added in the app subdirectory
add_executable(vizdoom_test "")
add_executable(mlpack_test "")

add_subdirectory(extern/ViZDoom)
add_subdirectory(app)

find_package(mlpack REQUIRED)

message(STATUS "mlpack include directory: ${MLPACK_INCLUDE_DIRS}")
message(STATUS "mlpack libraries: ${MLPACK_LIBRARIES}")

set_target_properties(vizdoom_test PROPERTIES LINK_DEPENDS_NO_SHARED true)
target_include_directories(vizdoom_test PRIVATE extern/ViZDoom/include ${MLPACK_INCLUDE_DIRS})
target_link_libraries(vizdoom_test PRIVATE libvizdoom_static ${MLPACK_LIBRARIES})

set_target_properties(mlpack_test PROPERTIES LINK_DEPENDS_NO_SHARED true)
target_include_directories(mlpack_test PRIVATE ${MLPACK_INCLUDE_DIRS})
target_link_libraries(mlpack_test PRIVATE ${MLPACK_LIBRARIES})




# Library example
#add_library(mylib STATIC mylib.cpp)
#target_include_directories(mylib PUBLIC include)
#add_library(another STATIC another.cpp another.h)
#target_link_libraries(another PUBLIC mylib)

# Setting Variables
#set(MY_VARIABLE "value")  # access via ${MY_VARIABLE}
