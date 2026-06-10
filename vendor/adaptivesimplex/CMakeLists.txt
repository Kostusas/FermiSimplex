cmake_minimum_required(VERSION 3.18...3.30)

project(
  adaptivesimplex
  VERSION 0.1.0
  LANGUAGES CXX
)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(ADAPTIVESIMPLEX_BUILD_EXAMPLES "Build AdaptiveSimplex examples" OFF)
option(ADAPTIVESIMPLEX_INSTALL "Install AdaptiveSimplex CMake package" ON)

if(NOT TARGET OpenMP::OpenMP_CXX)
  find_package(OpenMP QUIET COMPONENTS CXX)
endif()
set(ADAPTIVESIMPLEX_OPENMP_DEPENDENCY "")
if(OpenMP_CXX_FOUND)
  set(ADAPTIVESIMPLEX_OPENMP_DEPENDENCY "find_dependency(OpenMP COMPONENTS CXX)")
endif()

add_library(adaptivesimplex_core INTERFACE)
add_library(adaptivesimplex::core ALIAS adaptivesimplex_core)
target_compile_features(adaptivesimplex_core INTERFACE cxx_std_20)
target_include_directories(
  adaptivesimplex_core INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
set_target_properties(
  adaptivesimplex_core PROPERTIES
  EXPORT_NAME core
)

add_library(adaptivesimplex_adaptive INTERFACE)
add_library(adaptivesimplex::adaptive ALIAS adaptivesimplex_adaptive)
target_compile_features(adaptivesimplex_adaptive INTERFACE cxx_std_20)
target_include_directories(
  adaptivesimplex_adaptive INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(adaptivesimplex_adaptive INTERFACE adaptivesimplex::core)
if(OpenMP_CXX_FOUND)
  target_link_libraries(adaptivesimplex_adaptive INTERFACE OpenMP::OpenMP_CXX)
endif()
set_target_properties(
  adaptivesimplex_adaptive PROPERTIES
  EXPORT_NAME adaptive
)

if(BUILD_TESTING AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/core_smoke.cpp")
  add_executable(adaptivesimplex_core_smoke tests/core_smoke.cpp)
  target_compile_features(adaptivesimplex_core_smoke PRIVATE cxx_std_20)
  target_link_libraries(adaptivesimplex_core_smoke PRIVATE adaptivesimplex::core)
  add_test(NAME adaptivesimplex_core_smoke COMMAND adaptivesimplex_core_smoke)

  add_executable(adaptivesimplex_adaptive_smoke tests/adaptive_smoke.cpp)
  target_compile_features(adaptivesimplex_adaptive_smoke PRIVATE cxx_std_20)
  target_link_libraries(adaptivesimplex_adaptive_smoke PRIVATE adaptivesimplex::adaptive)
  add_test(NAME adaptivesimplex_adaptive_smoke COMMAND adaptivesimplex_adaptive_smoke)
endif()

if(ADAPTIVESIMPLEX_BUILD_EXAMPLES AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/scalar_integral.cpp")
  add_executable(adaptivesimplex_scalar_integral examples/scalar_integral.cpp)
  target_compile_features(adaptivesimplex_scalar_integral PRIVATE cxx_std_20)
  target_link_libraries(adaptivesimplex_scalar_integral PRIVATE adaptivesimplex::adaptive)

  add_executable(adaptivesimplex_vector_integral examples/vector_integral.cpp)
  target_compile_features(adaptivesimplex_vector_integral PRIVATE cxx_std_20)
  target_link_libraries(adaptivesimplex_vector_integral PRIVATE adaptivesimplex::adaptive)
endif()

if(ADAPTIVESIMPLEX_INSTALL)
  install(
    TARGETS adaptivesimplex_core adaptivesimplex_adaptive
    EXPORT adaptivesimplex-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  )
  install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )
  install(
    EXPORT adaptivesimplex-targets
    NAMESPACE adaptivesimplex::
    FILE adaptivesimplex-targets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/adaptivesimplex"
  )

  configure_package_config_file(
    cmake/adaptivesimplex-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/adaptivesimplex-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/adaptivesimplex"
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/adaptivesimplex-config-version.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
  )
  install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/adaptivesimplex-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/adaptivesimplex-config-version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/adaptivesimplex"
  )
endif()
