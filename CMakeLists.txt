cmake_minimum_required(VERSION 3.28...3.30)

project(
  fermisimplex
  VERSION 0.1.0
  LANGUAGES CXX
)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(FERMISIMPLEX_TOP_LEVEL_DEFAULT ON)
  include(CTest)
else()
  set(FERMISIMPLEX_TOP_LEVEL_DEFAULT OFF)
endif()

option(
  FERMISIMPLEX_BUILD_PYTHON
  "Build the fermisimplex Python extension"
  "${FERMISIMPLEX_TOP_LEVEL_DEFAULT}"
)
option(
  FERMISIMPLEX_BUILD_TESTING
  "Build the fermisimplex C++ tests"
  "${FERMISIMPLEX_TOP_LEVEL_DEFAULT}"
)

add_subdirectory(cpp)

if(FERMISIMPLEX_BUILD_PYTHON)
  add_subdirectory(python)
endif()
