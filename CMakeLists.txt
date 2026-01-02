cmake_minimum_required(VERSION 3.20)
project(fluxfish LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(liquid_chess
    cpp/src/main.cpp
    cpp/src/position.cpp
    cpp/src/uci/uci.cpp
)

target_include_directories(liquid_chess PRIVATE cpp/include)
