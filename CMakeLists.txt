cmake_minimum_required(VERSION 3.12)
project(basesvd)

set(CMAKE_CXX_STANDARD 17)
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g") # enable symbol generation
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z") # enable C++17 standard
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall") # enable all compiler warning flags
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3") # turn off optimizattions
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor") # enable Opem MP
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp") # enable Opem MP
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lomp") # enable Opem MP
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx") # auto-vectorization

# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lprofiler") # enable perf profiler
#

add_executable(svd_cpu svd_cpu.cpp matrix.h svd_serial.h timing.h)
