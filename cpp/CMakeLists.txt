find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)
find_package(LAPACK REQUIRED)
find_package(adaptivesimplex CONFIG QUIET)
if(NOT adaptivesimplex_FOUND)
  include(FetchContent)
  set(ADAPTIVESIMPLEX_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(ADAPTIVESIMPLEX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(ADAPTIVESIMPLEX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(ADAPTIVESIMPLEX_BUILD_DOC_TOOLS OFF CACHE BOOL "" FORCE)
  set(ADAPTIVESIMPLEX_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    adaptivesimplex
    GIT_REPOSITORY https://gitlab.kwant-project.org/qt/adaptivesimplex.git
    GIT_TAG d1e898190300f051e344f549d68534c059fd486b
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(adaptivesimplex)
endif()

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
foreach(_ltt_blas_symbol IN ITEMS zhemv_ zhemv ZHEMV)
  string(MAKE_C_IDENTIFIER "${_ltt_blas_symbol}" _ltt_blas_symbol_id)
  check_cxx_source_compiles(
    "extern \"C\" void ${_ltt_blas_symbol}(); int main() { auto fn = &${_ltt_blas_symbol}; return fn == nullptr; }"
    "LINEARTETRAHEDRON_HAS_BLAS_${_ltt_blas_symbol_id}"
  )
  if(LINEARTETRAHEDRON_HAS_BLAS_${_ltt_blas_symbol_id})
    set(LINEARTETRAHEDRON_BLAS_ZHEMV_SYMBOL "${_ltt_blas_symbol}")
    break()
  endif()
endforeach()
foreach(_ltt_blas_symbol IN ITEMS zgemv_ zgemv ZGEMV)
  string(MAKE_C_IDENTIFIER "${_ltt_blas_symbol}" _ltt_blas_symbol_id)
  check_cxx_source_compiles(
    "extern \"C\" void ${_ltt_blas_symbol}(); int main() { auto fn = &${_ltt_blas_symbol}; return fn == nullptr; }"
    "LINEARTETRAHEDRON_HAS_BLAS_${_ltt_blas_symbol_id}"
  )
  if(LINEARTETRAHEDRON_HAS_BLAS_${_ltt_blas_symbol_id})
    set(LINEARTETRAHEDRON_BLAS_ZGEMV_SYMBOL "${_ltt_blas_symbol}")
    break()
  endif()
endforeach()
set(CMAKE_REQUIRED_LIBRARIES "${_ltt_check_required_libraries}")
set(CMAKE_REQUIRED_LINK_OPTIONS "${_ltt_check_required_link_options}")

if(NOT LINEARTETRAHEDRON_LAPACK_ZHEEVD_SYMBOL)
  message(FATAL_ERROR "Could not find a callable LAPACK zheevd symbol")
endif()
if(NOT LINEARTETRAHEDRON_BLAS_ZHEMV_SYMBOL)
  message(FATAL_ERROR "Could not find a callable BLAS zhemv symbol")
endif()
if(NOT LINEARTETRAHEDRON_BLAS_ZGEMV_SYMBOL)
  message(FATAL_ERROR "Could not find a callable BLAS zgemv symbol")
endif()

set(
  _lineartetrahedron_sources
  src/bindings.cpp
  src/charge.cpp
  src/density.cpp
  src/integration_workspace.cpp
  src/linalg.cpp
  src/runtime.cpp
  src/vertex_spectra.cpp
  src/tight_binding.cpp
)

nanobind_add_module(
  _native
  NB_STATIC
  ${_lineartetrahedron_sources}
)
target_compile_features(_native PRIVATE cxx_std_20)
target_include_directories(_native PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(_native PRIVATE adaptivesimplex::adaptive adaptivesimplex::cut ${LAPACK_LIBRARIES})
target_link_options(_native PRIVATE ${LAPACK_LINKER_FLAGS})
target_compile_definitions(
  _native PRIVATE
  LINEARTETRAHEDRON_LAPACK_ZHEEVD=${LINEARTETRAHEDRON_LAPACK_ZHEEVD_SYMBOL}
  LINEARTETRAHEDRON_BLAS_ZHEMV=${LINEARTETRAHEDRON_BLAS_ZHEMV_SYMBOL}
  LINEARTETRAHEDRON_BLAS_ZGEMV=${LINEARTETRAHEDRON_BLAS_ZGEMV_SYMBOL}
)

install(TARGETS _native LIBRARY DESTINATION lineartetrahedron)
