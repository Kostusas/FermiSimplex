cmake_minimum_required(VERSION 3.18)

foreach(required_variable IN ITEMS
  PACKAGE_BUILD_DIR
  CONSUMER_SOURCE_DIR
  WORK_DIR
  GENERATOR
  CTEST_COMMAND
)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${required_variable} is required")
  endif()
endforeach()

function(run_checked description)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${description} failed with exit code ${result}")
  endif()
endfunction()

set(install_prefix "${WORK_DIR}/prefix")
set(consumer_build_dir "${WORK_DIR}/build")
file(REMOVE_RECURSE "${WORK_DIR}")

set(install_command
  "${CMAKE_COMMAND}" --install "${PACKAGE_BUILD_DIR}" --prefix "${install_prefix}"
)
if(NOT "${BUILD_CONFIG}" STREQUAL "")
  list(APPEND install_command --config "${BUILD_CONFIG}")
endif()
run_checked("Installing lineartetrahedron" ${install_command})

set(configure_command
  "${CMAKE_COMMAND}"
  -S "${CONSUMER_SOURCE_DIR}"
  -B "${consumer_build_dir}"
  -G "${GENERATOR}"
  "-DCMAKE_PREFIX_PATH=${install_prefix}"
)
if(NOT "${GENERATOR_PLATFORM}" STREQUAL "")
  list(APPEND configure_command -A "${GENERATOR_PLATFORM}")
endif()
if(NOT "${GENERATOR_TOOLSET}" STREQUAL "")
  list(APPEND configure_command -T "${GENERATOR_TOOLSET}")
endif()
if(NOT "${MAKE_PROGRAM}" STREQUAL "")
  list(APPEND configure_command "-DCMAKE_MAKE_PROGRAM=${MAKE_PROGRAM}")
endif()
if(NOT "${CXX_COMPILER}" STREQUAL "")
  list(APPEND configure_command "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}")
endif()
if(NOT "${TOOLCHAIN_FILE}" STREQUAL "")
  list(APPEND configure_command "-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}")
endif()
if(NOT "${BUILD_TYPE}" STREQUAL "")
  list(APPEND configure_command "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
endif()
if(
  NOT "${ADAPTIVESIMPLEX_DIR}" STREQUAL "" AND
  NOT "${ADAPTIVESIMPLEX_DIR}" MATCHES "-NOTFOUND$"
)
  list(APPEND configure_command "-Dadaptivesimplex_DIR=${ADAPTIVESIMPLEX_DIR}")
endif()
run_checked("Configuring the package consumer" ${configure_command})

set(build_command "${CMAKE_COMMAND}" --build "${consumer_build_dir}")
if(NOT "${BUILD_CONFIG}" STREQUAL "")
  list(APPEND build_command --config "${BUILD_CONFIG}")
endif()
run_checked("Building the package consumer" ${build_command})

set(test_command
  "${CTEST_COMMAND}" --test-dir "${consumer_build_dir}" --output-on-failure
)
if(NOT "${BUILD_CONFIG}" STREQUAL "")
  list(APPEND test_command --build-config "${BUILD_CONFIG}")
endif()
run_checked("Running the package consumer" ${test_command})
