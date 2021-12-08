cmake_minimum_required(VERSION 3.20)

project(file_reading_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
	main.cu
)

add_executable(file_reading_cuda ${SOURCES})