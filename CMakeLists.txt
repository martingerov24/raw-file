cmake_minimum_required(VERSION 3.20)

project(file_reading_cuda LANGUAGES CXX CUDA)
set(TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

find_package(OpenGL REQUIRED)
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

set(HEADERS 
    source/cudaManager.h
	source/stb_image.h
	source/stb_image_resize.h
)
set(SOURCES
	main.cu
    source/cudaManager.cu
    source/kernelnvoke.cu
	${IMGUI_SOURCES}
)

add_executable(${PROJECT_NAME} ${HEADERS} ${SOURCES})

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
)

target_link_libraries(${PROJECT_NAME} PUBLIC
	glad
    glfw
    ${OPENGL_LIBRARIES}
    ${CUDA_LIBS}
    ${CMAKE_DL_LIBS}
)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       >)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:MSVC>:
                       --MTR_ENABLED
                       >)