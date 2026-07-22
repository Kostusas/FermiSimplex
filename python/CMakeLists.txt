cmake_minimum_required(VERSION 3.18...3.30)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(
    fermisimplex_python
    VERSION 0.1.0
    LANGUAGES CXX
  )
endif()

if(NOT TARGET fermisimplex::core)
  find_package(fermisimplex CONFIG REQUIRED)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(
  _native
  NB_STATIC
  src/bindings/certification.cpp
  src/bindings/fermi_surface.cpp
  src/bindings/spectral_mesh.cpp
  src/bindings/integration_types.cpp
  src/bindings/module.cpp
)
target_compile_features(_native PRIVATE cxx_std_20)
target_include_directories(_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings")
target_link_libraries(_native PRIVATE fermisimplex::core)

install(TARGETS _native LIBRARY DESTINATION fermisimplex)
