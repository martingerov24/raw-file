cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_ARCHITECTURES 50)
project(cudaRawFile LANGUAGES CXX CUDA)

function(BUILD_PTX name)
    if (WIN32)
        set(NVCC "$ENV{CUDA_PATH}/bin/nvcc.exe")
    else()
        set(NVCC "$ENV{CUDA_PATH}/bin/nvcc")
    endif()

    set(NVCC_FLAGS
        -m64
        --use_fast_math
        -cudart static
        --gpu-architecture=compute_${CMAKE_CUDA_ARCHITECTURES}
        --gpu-code=sm_${CMAKE_CUDA_ARCHITECTURES},compute_${CMAKE_CUDA_ARCHITECTURES}
        -O4
    )
    set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cu)
    set(OUTPUT_FILE ${CMAKE_BINARY_DIR}/${name}.ptx)

    set(EXECUTE_COMMAND
        ${NVCC}
        ${NVCC_FLAGS}
        -ccbin ${CMAKE_CXX_COMPILER}
        ${GPU_INCLUDES}
        -ptx -o ${OUTPUT_FILE} ${SOURCE_FILE}
    )
    add_definitions(-DCUDA_INTEROP)
    add_custom_target(${name} ALL DEPENDS ${OUTPUT_FILE})

    add_custom_command(
        COMMAND    ${EXECUTE_COMMAND}
        DEPENDS    ${SOURCE_FILE}
        OUTPUT     ${OUTPUT_FILE}
    )
    install(FILES ${OUTPUT_FILE} DESTINATION ${INSTALL_PATH})
endfunction(BUILD_PTX)

if(DEFINED ENV{CUDA_PATH})
    set(INSTALL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build)
    set(CudaToolkitDir $ENV{CUDA_PATH})
    set(CUDA_INCLUDE_DIRS ${CudaToolkitDir}/include)
    set(CUDA_LIBS
        ${CudaToolkitDir}/lib/x64
        ${CudaToolkitDir}/lib64
        ${CudaToolkitDir}/lib64/stubs
    )
    build_ptx(kernelnvoke)
else()
	message(FATAL_ERROR "CUDA PATH not found in environment but is required.")
endif()

set(HEADERS
    cudaManager.h
)

set(DEVICE_SOURCES
    cudaManager.cu
    kernelnvoke.cu
)

set_source_files_properties(${HEADERS} ${SOURCES} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${DEVICE_SOURCES}     PROPERTIES LANGUAGE CUDA)

add_library(${PROJECT_NAME} STATIC ${HEADERS} ${DEVICE_SOURCES})

add_dependencies(${PROJECT_NAME} kernelnvoke)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty
)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    --use_fast_math
    >
)

target_link_directories(${PROJECT_NAME} PUBLIC
    ${CUDA_LIBS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    cuda
    cudart
)

install(TARGETS ${PROJECT_NAME} DESTINATION ${INSTALL_PATH})
