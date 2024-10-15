
cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(slp C)

add_executable(
  ${CMAKE_PROJECT_NAME}
  test.c
  slp.c
)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -g -O0)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE -lm)

