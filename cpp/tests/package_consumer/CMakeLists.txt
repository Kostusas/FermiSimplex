cmake_minimum_required(VERSION 3.18)

project(fermisimplex_package_consumer LANGUAGES CXX)

enable_testing()

find_package(fermisimplex CONFIG REQUIRED)

add_executable(fermisimplex_package_consumer main.cpp)
target_compile_features(fermisimplex_package_consumer PRIVATE cxx_std_20)
target_link_libraries(
  fermisimplex_package_consumer PRIVATE
  fermisimplex::core
)
add_test(
  NAME fermisimplex_package_consumer_smoke
  COMMAND fermisimplex_package_consumer
)
