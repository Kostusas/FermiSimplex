cmake_minimum_required(VERSION 3.28...3.30)

set(LINEARTETRAHEDRON_CPP_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(
    lineartetrahedron
    VERSION 0.1.0
    LANGUAGES CXX
  )
  set(LINEARTETRAHEDRON_CPP_TOP_LEVEL ON)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(
  LINEARTETRAHEDRON_BUILD_TESTING
  "Build the lineartetrahedron C++ tests"
  "${LINEARTETRAHEDRON_CPP_TOP_LEVEL}"
)
option(
  LINEARTETRAHEDRON_BUILD_BENCHMARKS
  "Build the lineartetrahedron C++ benchmarks"
  OFF
)
option(
  LINEARTETRAHEDRON_INSTALL_CPP
  "Install the lineartetrahedron C++ library and package metadata"
  "${LINEARTETRAHEDRON_CPP_TOP_LEVEL}"
)

find_package(LAPACK REQUIRED)
if(NOT TARGET adaptivesimplex::adaptive)
  find_package(adaptivesimplex CONFIG QUIET)
endif()
if(NOT TARGET adaptivesimplex::adaptive)
  include(FetchContent)
  set(ADAPTIVESIMPLEX_BUILD_TESTING OFF)
  set(ADAPTIVESIMPLEX_BUILD_EXAMPLES OFF)
  set(ADAPTIVESIMPLEX_BUILD_BENCHMARKS OFF)
  set(ADAPTIVESIMPLEX_BUILD_DOC_TOOLS OFF)
  set(ADAPTIVESIMPLEX_ENABLE_OPENMP OFF)
  set(_ltt_adaptivesimplex_fetch_options)
  if(NOT LINEARTETRAHEDRON_INSTALL_CPP)
    list(APPEND _ltt_adaptivesimplex_fetch_options EXCLUDE_FROM_ALL)
  endif()
  FetchContent_Declare(
    adaptivesimplex
    GIT_REPOSITORY https://gitlab.kwant-project.org/qt/adaptivesimplex.git
    GIT_TAG d1e898190300f051e344f549d68534c059fd486b
    ${_ltt_adaptivesimplex_fetch_options}
  )
  FetchContent_MakeAvailable(adaptivesimplex)
endif()

include(CheckCXXSourceCompiles)

set(_ltt_check_required_libraries "${CMAKE_REQUIRED_LIBRARIES}")
set(_ltt_check_required_link_options "${CMAKE_REQUIRED_LINK_OPTIONS}")
set(CMAKE_REQUIRED_LIBRARIES LAPACK::LAPACK)
set(CMAKE_REQUIRED_LINK_OPTIONS)

function(_lineartetrahedron_find_fortran_symbol output check_name)
  foreach(_ltt_symbol IN LISTS ARGN)
    string(MAKE_C_IDENTIFIER "${_ltt_symbol}" _ltt_symbol_id)
    set(_ltt_check "LINEARTETRAHEDRON_HAS_${check_name}_${_ltt_symbol_id}")
    check_cxx_source_compiles(
      "extern \"C\" void ${_ltt_symbol}(); int main() { auto fn = &${_ltt_symbol}; return fn == nullptr; }"
      "${_ltt_check}"
    )
    if(${_ltt_check})
      set(${output} "${_ltt_symbol}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

_lineartetrahedron_find_fortran_symbol(
  LINEARTETRAHEDRON_LAPACK_ZHEEVD_SYMBOL LAPACK_ZHEEVD zheevd_ zheevd ZHEEVD
)
_lineartetrahedron_find_fortran_symbol(
  LINEARTETRAHEDRON_LAPACK_ZPOTRF_SYMBOL LAPACK_ZPOTRF zpotrf_ zpotrf ZPOTRF
)
_lineartetrahedron_find_fortran_symbol(
  LINEARTETRAHEDRON_BLAS_ZGEMM_SYMBOL BLAS_ZGEMM zgemm_ zgemm ZGEMM
)
_lineartetrahedron_find_fortran_symbol(
  LINEARTETRAHEDRON_BLAS_ZHER2K_SYMBOL BLAS_ZHER2K zher2k_ zher2k ZHER2K
)
_lineartetrahedron_find_fortran_symbol(
  LINEARTETRAHEDRON_BLAS_ZHEMM_SYMBOL BLAS_ZHEMM zhemm_ zhemm ZHEMM
)
set(CMAKE_REQUIRED_LIBRARIES "${_ltt_check_required_libraries}")
set(CMAKE_REQUIRED_LINK_OPTIONS "${_ltt_check_required_link_options}")

if(NOT LINEARTETRAHEDRON_LAPACK_ZHEEVD_SYMBOL)
  message(FATAL_ERROR "Could not find a callable LAPACK zheevd symbol")
endif()
if(NOT LINEARTETRAHEDRON_LAPACK_ZPOTRF_SYMBOL)
  message(FATAL_ERROR "Could not find a callable LAPACK zpotrf symbol")
endif()
if(NOT LINEARTETRAHEDRON_BLAS_ZGEMM_SYMBOL)
  message(FATAL_ERROR "Could not find a callable BLAS zgemm symbol")
endif()
if(NOT LINEARTETRAHEDRON_BLAS_ZHER2K_SYMBOL)
  message(FATAL_ERROR "Could not find a callable BLAS zher2k symbol")
endif()
if(NOT LINEARTETRAHEDRON_BLAS_ZHEMM_SYMBOL)
  message(FATAL_ERROR "Could not find a callable BLAS zhemm symbol")
endif()

set(
  _lineartetrahedron_core_sources
  src/core/spectral_mesh.cpp
  src/core/tight_binding.cpp
  src/linalg/blas.cpp
  src/linalg/lapack.cpp
  src/certification/bounds/mu_bounds.cpp
  src/certification/bounds/occupation_bounds.cpp
  src/certification/linalg/cholesky.cpp
  src/certification/linalg/rotated_blocks.cpp
  src/certification/simplex/anchor_selection.cpp
  src/certification/simplex/occupation_certificate.cpp
  src/certification/simplex/simplex_blocks.cpp
  src/certification/simplex/vertex_blocks.cpp
  src/certification/simplex_certificate.cpp
  src/integration/charge.cpp
  src/integration/density.cpp
  src/integration/integration.cpp
  src/integration/projected_error.cpp
  src/fermi_surface/adaptive_refinement.cpp
  src/fermi_surface/fermi_surface.cpp
  src/fermi_surface/simplex_classification.cpp
  src/fermi_surface/surface_extraction.cpp
  src/fermi_surface/surface_triangulation.cpp
)

add_library(lineartetrahedron_core STATIC ${_lineartetrahedron_core_sources})
add_library(lineartetrahedron::core ALIAS lineartetrahedron_core)
target_compile_features(lineartetrahedron_core PUBLIC cxx_std_20)
target_include_directories(
  lineartetrahedron_core
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
set_target_properties(
  lineartetrahedron_core PROPERTIES
  EXPORT_NAME core
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(
  lineartetrahedron_core
  PUBLIC
    adaptivesimplex::adaptive
  PRIVATE
    adaptivesimplex::cut
    LAPACK::LAPACK
)
target_compile_definitions(
  lineartetrahedron_core PRIVATE
  LINEARTETRAHEDRON_LAPACK_ZHEEVD=${LINEARTETRAHEDRON_LAPACK_ZHEEVD_SYMBOL}
  LINEARTETRAHEDRON_LAPACK_ZPOTRF=${LINEARTETRAHEDRON_LAPACK_ZPOTRF_SYMBOL}
  LINEARTETRAHEDRON_BLAS_ZGEMM=${LINEARTETRAHEDRON_BLAS_ZGEMM_SYMBOL}
  LINEARTETRAHEDRON_BLAS_ZHER2K=${LINEARTETRAHEDRON_BLAS_ZHER2K_SYMBOL}
  LINEARTETRAHEDRON_BLAS_ZHEMM=${LINEARTETRAHEDRON_BLAS_ZHEMM_SYMBOL}
)

if(LINEARTETRAHEDRON_BUILD_TESTING)
  enable_testing()

  set(_ltt_cpp_test_targets)
  foreach(_ltt_test IN ITEMS
    occupation_bounds
    certificate_validation
    simplex_certificate
    spectral_mesh
    projected_error
    charge_certificates
  )
    set(_ltt_test_target "lineartetrahedron_${_ltt_test}_tests")
    add_executable(
      ${_ltt_test_target}
      "tests/test_${_ltt_test}.cpp"
    )
    target_compile_features(${_ltt_test_target} PRIVATE cxx_std_20)
    target_include_directories(
      ${_ltt_test_target} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    )
    target_link_libraries(${_ltt_test_target} PRIVATE lineartetrahedron::core)
    add_test(
      NAME ${_ltt_test_target}
      COMMAND $<TARGET_FILE:${_ltt_test_target}>
    )
    list(APPEND _ltt_cpp_test_targets ${_ltt_test_target})
  endforeach()
  add_custom_target(lineartetrahedron_cpp_tests DEPENDS ${_ltt_cpp_test_targets})

  if(LINEARTETRAHEDRON_INSTALL_CPP)
    add_test(
      NAME lineartetrahedron_installed_package_consumer
      COMMAND
        "${CMAKE_COMMAND}"
        "-DPACKAGE_BUILD_DIR=${CMAKE_BINARY_DIR}"
        "-DCONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/tests/package_consumer"
        "-DWORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/package_consumer"
        "-DGENERATOR=${CMAKE_GENERATOR}"
        "-DGENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
        "-DGENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
        "-DMAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
        "-DCXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DTOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
        "-DBUILD_TYPE=${CMAKE_BUILD_TYPE}"
        "-DBUILD_CONFIG=$<CONFIG>"
        "-DADAPTIVESIMPLEX_DIR=${adaptivesimplex_DIR}"
        "-DCTEST_COMMAND=${CMAKE_CTEST_COMMAND}"
        -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/package_consumer/verify.cmake"
    )
    set_tests_properties(
      lineartetrahedron_installed_package_consumer PROPERTIES
      LABELS package
    )
  endif()
endif()

if(LINEARTETRAHEDRON_BUILD_BENCHMARKS)
  add_executable(
    lineartetrahedron_occupation_bounds_benchmark
    tools/benchmark_occupation_bounds.cpp
  )
  target_compile_features(lineartetrahedron_occupation_bounds_benchmark PRIVATE cxx_std_20)
  target_include_directories(
    lineartetrahedron_occupation_bounds_benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(
    lineartetrahedron_occupation_bounds_benchmark PRIVATE
    lineartetrahedron::core
  )
endif()

if(LINEARTETRAHEDRON_INSTALL_CPP)
  install(
    TARGETS lineartetrahedron_core
    EXPORT lineartetrahedron-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  )
  install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )
  install(
    EXPORT lineartetrahedron-targets
    NAMESPACE lineartetrahedron::
    FILE lineartetrahedron-targets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/lineartetrahedron"
  )

  configure_package_config_file(
    cmake/lineartetrahedron-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/lineartetrahedron-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/lineartetrahedron"
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/lineartetrahedron-config-version.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMinorVersion
  )
  install(
    FILES
      "${CMAKE_CURRENT_BINARY_DIR}/lineartetrahedron-config.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/lineartetrahedron-config-version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/lineartetrahedron"
  )
endif()
