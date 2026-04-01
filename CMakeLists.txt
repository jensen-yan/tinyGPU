cmake_minimum_required(VERSION 3.16)

project(tinyGPU LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(tinygpu
    src/main.cpp
    src/simulator.cpp
)

target_include_directories(tinygpu PRIVATE include)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(tinygpu PRIVATE -Wall -Wextra -Wpedantic)
endif()
