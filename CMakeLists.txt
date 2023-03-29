cmake_minimum_required(VERSION 3.20)
project(file_reading_cuda LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)

if(DEFINED ENV{CUDA_PATH})
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})
    set(CUDA_INCLUDE_DIRS
        ${CUDA_TOOLKIT_ROOT_DIR}/include
        "./source/"
    )
    add_subdirectory(source build)
    set(CUDA_LINK_LIBS
        cudaRawFile
    )
    set(CUDA_LIBS_DIRS
        $ENV{CUDA_MATH_LIBS_PATH}
        $ENV{CUDNN_PATH}/lib64
        $ENV{TRT_PATH}/lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64/
    )
    set(CUDA_LINK_RT cudart)
    message("-- CUDA_PATH = ${CUDA_TOOLKIT_ROOT_DIR}")
else()
    message("-- CUDA_PATH is not set")
endif()


#-----------imgui Dependencies----------
set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/imgui")
set(IMGUI_SOURCES
    "${IMGUI_DIR}/imgui_demo.cpp"
    "${IMGUI_DIR}/imgui_draw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_glfw.h"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
    "${IMGUI_DIR}/backends/imgui_impl_opengl3.h"
    "${IMGUI_DIR}/imgui_tables.cpp"
    "${IMGUI_DIR}/imgui_widgets.cpp"
    "${IMGUI_DIR}/imgui.cpp"
)

set( imgui_impl_files
    ${IMGUI_DIR}/examples/imgui_impl_glfw.h
    ${IMGUI_DIR}/examples/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/examples/imgui_impl_opengl3.h
    ${IMGUI_DIR}/examples/imgui_impl_opengl3.cpp
)

#-----------glad Dependencies-----------
set(GLAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glad")
add_library("glad" STATIC "${GLAD_DIR}/src/glad.c")
target_include_directories("glad" PRIVATE "${GLAD_DIR}/include")

#-----------glfw Dependencies-----------
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glfw/)
set(GLFW_LIBRARY_TYPE STATIC)
add_subdirectory(${GLFW_DIR})
  
set(SOURCES
	main.cpp
	thirdparty/stb_image.h
	thirdparty/stb_image_resize.h
	${IMGUI_SOURCES}
)

add_executable(${PROJECT_NAME} ${SOURCES})

message(STATUS "====== CUDA Located: ${CUDA_INCLUDE_DIRS}")
message(STATUS "====== IMGUI Located: ${IMGUI_DIR}")
message(STATUS "====== OpenGL Located: ${OPENGL_LIBRARIES}")

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CUDA_INCLUDE_DIRS}
	${GLAD_DIR}/include
    ${GLFW_DIR}/include
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends 
    ${CMAKE_CURRENT_SOURCE_DIR}/source
    ${CMAKE_CURRENT_SOURCE_DIR}/minitrace
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${OPENGL_LIBRARIES}
    ${CUDA_LINK_LIBS}
    ${CMAKE_DL_LIBS}
    ${CUDA_LINK_RT}
    glfw
	glad
)

target_link_directories(${PROJECT_NAME} PUBLIC
    ${CUDA_LIBS_DIRS}
)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       >)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:MSVC>:
                       --MTR_ENABLED
                       >)
