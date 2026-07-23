cmake_minimum_required(VERSION 3.28...3.30)

set(FERMISIMPLEX_CPP_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(
    FermiSimplex
    VERSION 0.1.0
    LANGUAGES CXX
  )
  set(FERMISIMPLEX_CPP_TOP_LEVEL ON)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(
  FERMISIMPLEX_BUILD_TESTING
  "Build the FermiSimplex C++ tests"
  "${FERMISIMPLEX_CPP_TOP_LEVEL}"
)
option(
  FERMISIMPLEX_BUILD_BENCHMARKS
  "Build the FermiSimplex C++ benchmarks"
  OFF
)
option(
  FERMISIMPLEX_INSTALL_CPP
  "Install the FermiSimplex C++ library and package metadata"
  "${FERMISIMPLEX_CPP_TOP_LEVEL}"
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
  set(_fs_adaptivesimplex_fetch_options)
  if(NOT FERMISIMPLEX_INSTALL_CPP)
    list(APPEND _fs_adaptivesimplex_fetch_options EXCLUDE_FROM_ALL)
  endif()
  FetchContent_Declare(
    adaptivesimplex
    GIT_REPOSITORY https://gitlab.kwant-project.org/qt/adaptivesimplex.git
    GIT_TAG d1e898190300f051e344f549d68534c059fd486b
    ${_fs_adaptivesimplex_fetch_options}
  )
  FetchContent_MakeAvailable(adaptivesimplex)
endif()

include(CheckCXXSourceCompiles)

set(_fs_check_required_libraries "${CMAKE_REQUIRED_LIBRARIES}")
set(_fs_check_required_link_options "${CMAKE_REQUIRED_LINK_OPTIONS}")
set(CMAKE_REQUIRED_LIBRARIES LAPACK::LAPACK)
set(CMAKE_REQUIRED_LINK_OPTIONS)

function(_fermisimplex_find_fortran_symbol output check_name)
  foreach(_fs_symbol IN LISTS ARGN)
    string(MAKE_C_IDENTIFIER "${_fs_symbol}" _fs_symbol_id)
    set(_fs_check "FERMISIMPLEX_HAS_${check_name}_${_fs_symbol_id}")
    check_cxx_source_compiles(
      "extern \"C\" void ${_fs_symbol}(); int main() { auto fn = &${_fs_symbol}; return fn == nullptr; }"
      "${_fs_check}"
    )
    if(${_fs_check})
      set(${output} "${_fs_symbol}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

_fermisimplex_find_fortran_symbol(
  FERMISIMPLEX_LAPACK_ZHEEVD_SYMBOL LAPACK_ZHEEVD zheevd_ zheevd ZHEEVD
)
_fermisimplex_find_fortran_symbol(
  FERMISIMPLEX_LAPACK_ZPOTRF_SYMBOL LAPACK_ZPOTRF zpotrf_ zpotrf ZPOTRF
)
_fermisimplex_find_fortran_symbol(
  FERMISIMPLEX_BLAS_ZGEMM_SYMBOL BLAS_ZGEMM zgemm_ zgemm ZGEMM
)
_fermisimplex_find_fortran_symbol(
  FERMISIMPLEX_BLAS_ZHER2K_SYMBOL BLAS_ZHER2K zher2k_ zher2k ZHER2K
)
_fermisimplex_find_fortran_symbol(
  FERMISIMPLEX_BLAS_ZHEMM_SYMBOL BLAS_ZHEMM zhemm_ zhemm ZHEMM
)
set(CMAKE_REQUIRED_LIBRARIES "${_fs_check_required_libraries}")
set(CMAKE_REQUIRED_LINK_OPTIONS "${_fs_check_required_link_options}")

if(NOT FERMISIMPLEX_LAPACK_ZHEEVD_SYMBOL)
  message(FATAL_ERROR "Could not find a callable LAPACK zheevd symbol")
endif()
if(NOT FERMISIMPLEX_LAPACK_ZPOTRF_SYMBOL)
  message(FATAL_ERROR "Could not find a callable LAPACK zpotrf symbol")
endif()
if(NOT FERMISIMPLEX_BLAS_ZGEMM_SYMBOL)
  message(FATAL_ERROR "Could not find a callable BLAS zgemm symbol")
endif()
if(NOT FERMISIMPLEX_BLAS_ZHER2K_SYMBOL)
  message(FATAL_ERROR "Could not find a callable BLAS zher2k symbol")
endif()
if(NOT FERMISIMPLEX_BLAS_ZHEMM_SYMBOL)
  message(FATAL_ERROR "Could not find a callable BLAS zhemm symbol")
endif()

set(
  _fermisimplex_core_sources
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

add_library(fermisimplex_core STATIC ${_fermisimplex_core_sources})
add_library(fermisimplex::core ALIAS fermisimplex_core)
target_compile_features(fermisimplex_core PUBLIC cxx_std_20)
target_include_directories(
  fermisimplex_core
  PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)
set_target_properties(
  fermisimplex_core PROPERTIES
  EXPORT_NAME core
  POSITION_INDEPENDENT_CODE ON
)
target_link_libraries(
  fermisimplex_core
  PUBLIC
    adaptivesimplex::adaptive
  PRIVATE
    adaptivesimplex::cut
    LAPACK::LAPACK
)
target_compile_definitions(
  fermisimplex_core PRIVATE
  FERMISIMPLEX_LAPACK_ZHEEVD=${FERMISIMPLEX_LAPACK_ZHEEVD_SYMBOL}
  FERMISIMPLEX_LAPACK_ZPOTRF=${FERMISIMPLEX_LAPACK_ZPOTRF_SYMBOL}
  FERMISIMPLEX_BLAS_ZGEMM=${FERMISIMPLEX_BLAS_ZGEMM_SYMBOL}
  FERMISIMPLEX_BLAS_ZHER2K=${FERMISIMPLEX_BLAS_ZHER2K_SYMBOL}
  FERMISIMPLEX_BLAS_ZHEMM=${FERMISIMPLEX_BLAS_ZHEMM_SYMBOL}
)

if(FERMISIMPLEX_BUILD_TESTING)
  enable_testing()

  set(_fs_cpp_test_targets)
  foreach(_fs_test IN ITEMS
    occupation_bounds
    certificate_validation
    simplex_certificate
    spectral_mesh
    projected_error
    charge_certificates
  )
    set(_fs_test_target "fermisimplex_${_fs_test}_tests")
    add_executable(
      ${_fs_test_target}
      "tests/test_${_fs_test}.cpp"
    )
    target_compile_features(${_fs_test_target} PRIVATE cxx_std_20)
    target_include_directories(
      ${_fs_test_target} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/tests"
    )
    target_link_libraries(${_fs_test_target} PRIVATE fermisimplex::core)
    add_test(
      NAME ${_fs_test_target}
      COMMAND $<TARGET_FILE:${_fs_test_target}>
    )
    list(APPEND _fs_cpp_test_targets ${_fs_test_target})
  endforeach()
  add_custom_target(fermisimplex_cpp_tests DEPENDS ${_fs_cpp_test_targets})

  if(FERMISIMPLEX_INSTALL_CPP)
    add_test(
      NAME fermisimplex_installed_package_consumer
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
      fermisimplex_installed_package_consumer PROPERTIES
      LABELS package
    )
  endif()
endif()

if(FERMISIMPLEX_BUILD_BENCHMARKS)
  find_package(Git QUIET)
  set(_fs_benchmark_git_commit "unknown")
  set(_fs_benchmark_git_dirty "unknown")
  if(Git_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      RESULT_VARIABLE _fs_benchmark_git_result
      OUTPUT_VARIABLE _fs_benchmark_git_output
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_fs_benchmark_git_result EQUAL 0)
      set(_fs_benchmark_git_commit "${_fs_benchmark_git_output}")
    endif()
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" status --porcelain
      WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
      RESULT_VARIABLE _fs_benchmark_status_result
      OUTPUT_VARIABLE _fs_benchmark_status_output
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    if(_fs_benchmark_status_result EQUAL 0)
      if(_fs_benchmark_status_output)
        set(_fs_benchmark_git_dirty "true")
      else()
        set(_fs_benchmark_git_dirty "false")
      endif()
    endif()
  endif()
  string(REPLACE ";" "|" _fs_benchmark_lapack "${LAPACK_LIBRARIES}")

  add_executable(
    fermisimplex_occupation_bounds_benchmark
    tools/benchmark_occupation_bounds.cpp
  )
  target_compile_features(fermisimplex_occupation_bounds_benchmark PRIVATE cxx_std_20)
  target_include_directories(
    fermisimplex_occupation_bounds_benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_link_libraries(
    fermisimplex_occupation_bounds_benchmark PRIVATE
    fermisimplex::core
  )

  add_executable(
    fermisimplex_performance_benchmark
    tools/benchmark_performance.cpp
  )
  target_compile_features(fermisimplex_performance_benchmark PRIVATE cxx_std_20)
  target_include_directories(
    fermisimplex_performance_benchmark PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
  )
  target_compile_definitions(
    fermisimplex_performance_benchmark PRIVATE
    FERMISIMPLEX_LAPACK_ZHEEVD=${FERMISIMPLEX_LAPACK_ZHEEVD_SYMBOL}
    FERMISIMPLEX_BENCHMARK_GIT_COMMIT="${_fs_benchmark_git_commit}"
    FERMISIMPLEX_BENCHMARK_GIT_DIRTY="${_fs_benchmark_git_dirty}"
    FERMISIMPLEX_BENCHMARK_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
    FERMISIMPLEX_BENCHMARK_BUILD_TYPE="$<CONFIG>"
    FERMISIMPLEX_BENCHMARK_SYSTEM="${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}"
    FERMISIMPLEX_BENCHMARK_LAPACK="${_fs_benchmark_lapack}"
  )
  target_link_libraries(
    fermisimplex_performance_benchmark PRIVATE
    fermisimplex::core
    LAPACK::LAPACK
  )

  add_custom_target(
    fermisimplex_run_performance_benchmark
    COMMAND
      "${CMAKE_COMMAND}" -E env
      OMP_NUM_THREADS=1
      OPENBLAS_NUM_THREADS=1
      MKL_NUM_THREADS=1
      VECLIB_MAXIMUM_THREADS=1
      "$<TARGET_FILE:fermisimplex_performance_benchmark>"
      --preset ci
      --output "${CMAKE_BINARY_DIR}/benchmarks/fermisimplex-performance.json"
    DEPENDS fermisimplex_performance_benchmark
    USES_TERMINAL
    VERBATIM
  )
endif()

if(FERMISIMPLEX_INSTALL_CPP)
  install(
    TARGETS fermisimplex_core
    EXPORT fermisimplex-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  )
  install(
    DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  )
  install(
    EXPORT fermisimplex-targets
    NAMESPACE fermisimplex::
    FILE fermisimplex-targets.cmake
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/fermisimplex"
  )

  configure_package_config_file(
    cmake/fermisimplex-config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/fermisimplex-config.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/fermisimplex"
  )
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/fermisimplex-config-version.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMinorVersion
  )
  install(
    FILES
      "${CMAKE_CURRENT_BINARY_DIR}/fermisimplex-config.cmake"
      "${CMAKE_CURRENT_BINARY_DIR}/fermisimplex-config-version.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/fermisimplex"
  )
endif()
