find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(adaptivesimplex CONFIG REQUIRED)
find_package(LAPACK REQUIRED)
if(APPLE AND DEFINED ENV{CONDA_PREFIX})
  set(_lineartetrahedron_conda_libomp "$ENV{CONDA_PREFIX}/lib/libomp.dylib")
  if(EXISTS "${_lineartetrahedron_conda_libomp}")
    set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "OpenMP CXX library names" FORCE)
    set(OpenMP_omp_LIBRARY "${_lineartetrahedron_conda_libomp}" CACHE FILEPATH "OpenMP libomp path" FORCE)
  endif()
endif()
find_package(OpenMP REQUIRED)

include(CheckCXXSourceCompiles)

set(_ltt_check_required_libraries "${CMAKE_REQUIRED_LIBRARIES}")
set(_ltt_check_required_link_options "${CMAKE_REQUIRED_LINK_OPTIONS}")
set(CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
set(CMAKE_REQUIRED_LINK_OPTIONS ${LAPACK_LINKER_FLAGS})
foreach(_ltt_lapack_symbol IN ITEMS zheevd_ zheevd ZHEEVD)
  string(MAKE_C_IDENTIFIER "${_ltt_lapack_symbol}" _ltt_lapack_symbol_id)
  check_cxx_source_compiles(
    "extern \"C\" void ${_ltt_lapack_symbol}(); int main() { auto fn = &${_ltt_lapack_symbol}; return fn == nullptr; }"
    "LINEARTETRAHEDRON_HAS_LAPACK_${_ltt_lapack_symbol_id}"
  )
  if(LINEARTETRAHEDRON_HAS_LAPACK_${_ltt_lapack_symbol_id})
    set(LINEARTETRAHEDRON_LAPACK_ZHEEVD_SYMBOL "${_ltt_lapack_symbol}")
    break()
  endif()
endforeach()
set(CMAKE_REQUIRED_LIBRARIES "${_ltt_check_required_libraries}")
set(CMAKE_REQUIRED_LINK_OPTIONS "${_ltt_check_required_link_options}")

if(NOT LINEARTETRAHEDRON_LAPACK_ZHEEVD_SYMBOL)
  message(FATAL_ERROR "Could not find a callable LAPACK zheevd symbol")
endif()

set(
  _lineartetrahedron_sources
  src/band_weights.cpp
  src/bindings.cpp
  src/charge_integrand.cpp
  src/density_integrand.cpp
  src/integration_state.cpp
  src/runtime.cpp
  src/simplex_rules.cpp
  src/vertex_spectra.cpp
  src/tight_binding.cpp
  src/types.cpp
)

nanobind_add_module(
  _native
  NB_STATIC
  ${_lineartetrahedron_sources}
)
target_compile_features(_native PRIVATE cxx_std_20)
target_include_directories(_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(_native PRIVATE adaptivesimplex::adaptive OpenMP::OpenMP_CXX ${LAPACK_LIBRARIES})
target_link_options(_native PRIVATE ${LAPACK_LINKER_FLAGS})
target_compile_definitions(
  _native PRIVATE
  LINEARTETRAHEDRON_LAPACK_ZHEEVD=${LINEARTETRAHEDRON_LAPACK_ZHEEVD_SYMBOL}
)

install(TARGETS _native LIBRARY DESTINATION lineartetrahedron)
