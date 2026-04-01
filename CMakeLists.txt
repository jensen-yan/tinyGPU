cmake_minimum_required(VERSION 3.16)

project(tinyGPU LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_library(tinygpu_core
    src/simulator.cpp
    src/kernels.cpp
)

target_include_directories(tinygpu_core PUBLIC include)

add_executable(tinygpu
    src/main.cpp
)

target_link_libraries(tinygpu PRIVATE tinygpu_core)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(tinygpu_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(tinygpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

find_package(GTest REQUIRED)

add_executable(tinygpu_tests
    tests/simulator_test.cpp
)

target_link_libraries(tinygpu_tests
    PRIVATE
        tinygpu_core
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tinygpu_tests)
