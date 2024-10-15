
cmake_minimum_required(VERSION 3.22)

project(slp C)

add_executable(
  ${CMAKE_PROJECT_NAME}
  test.c
  slp.c
)

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -g -O0)

