cmake_minimum_required(VERSION 3.20)
project(file_reading_cuda LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{CUDA_PATH})
    message(FATAL_ERROR "-- CUDA_PATH is not set")
endif()

add_subdirectory(source)
add_subdirectory(renderer)

set(SOURCES
	main.cpp
)
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/minitrace
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    cudaRawFile
    renderer
    ${CMAKE_DL_LIBS}
)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       >)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:MSVC>:
                       --MTR_ENABLED
                       >)
