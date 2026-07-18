cmake_minimum_required(VERSION 3.18)

project(lineartetrahedron_package_consumer LANGUAGES CXX)

enable_testing()

find_package(lineartetrahedron CONFIG REQUIRED)

add_executable(lineartetrahedron_package_consumer main.cpp)
target_compile_features(lineartetrahedron_package_consumer PRIVATE cxx_std_20)
target_link_libraries(
  lineartetrahedron_package_consumer PRIVATE
  lineartetrahedron::core
)
add_test(
  NAME lineartetrahedron_package_consumer_smoke
  COMMAND lineartetrahedron_package_consumer
)
