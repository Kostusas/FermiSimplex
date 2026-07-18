cmake_minimum_required(VERSION 3.28...3.30)

project(
  lineartetrahedron
  VERSION 0.1.0
  LANGUAGES CXX
)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(LINEARTETRAHEDRON_TOP_LEVEL_DEFAULT ON)
  include(CTest)
else()
  set(LINEARTETRAHEDRON_TOP_LEVEL_DEFAULT OFF)
endif()

option(
  LINEARTETRAHEDRON_BUILD_PYTHON
  "Build the lineartetrahedron Python extension"
  "${LINEARTETRAHEDRON_TOP_LEVEL_DEFAULT}"
)
option(
  LINEARTETRAHEDRON_BUILD_TESTING
  "Build the lineartetrahedron C++ tests"
  "${LINEARTETRAHEDRON_TOP_LEVEL_DEFAULT}"
)

add_subdirectory(cpp)

if(LINEARTETRAHEDRON_BUILD_PYTHON)
  add_subdirectory(python)
endif()
