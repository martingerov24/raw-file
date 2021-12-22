cmake_minimum_required(VERSION 3.20)

project(file_reading_cuda LANGUAGES CXX CUDA)
set(TOOLKIT_ROOT_DIR $ENV{CUDA_PATH})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#-----------OpenCV Dependancies---------
set(OPENCV_DIR D:/marting/opencv-feature-matching/opencv/Build/install)
find_package(OpenCV CONFIG REQUIRED PATHS ${OPENCV_DIR}) 
set(OpenCV_LIBS opencv_core454 opencv_highgui454 opencv_videoio454 opencv_imgcodecs454 opencv_imgproc454 opencv_video454 opencv_xfeatures2d454 opencv_features2d454 opencv_cudafeatures2d454 opencv_calib3d454)

 
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
set(GLFW_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/glfw/")
set(GLFW_LIBRARY_TYPE STATIC)
add_subdirectory(${GLFW_DIR})

set(SOURCES
	main.cu
	${IMGUI_SOURCES}
	Keypoint.cu
	CudaClass.h
	Cuda.cu
)

add_executable(file_reading_cuda ${SOURCES})

message(STATUS "====== OpenCV Located: ${OPENCV_PATH}")
message(STATUS "====== OpenCV LIBS: ${OpenCV_LIBS}")
message(STATUS "====== CUDA Located: ${CUDA_INCLUDE_DIRS}")
message(STATUS "====== IMGUI Located: ${IMGUI_DIR}")
message(STATUS "====== OpenGL Located: ${OPENGL_LIBRARIES}")

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CUDA_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	"${GLAD_DIR}/include"
    "${GLFW_DIR}/include"
    "${IMGUI_DIR}"
    "${IMGUI_DIR}/backends" 
)
target_link_directories(${PROJECT_NAME} PRIVATE
	${OPENCV_DIR}/x64/vc16/lib
)
target_link_libraries(${PROJECT_NAME} PUBLIC
	glad
    glfw
    ${OPENGL_LIBRARIES}
    ${OpenCV_LIBS}
    ${CUDA_LIBS}
    ${CMAKE_DL_LIBS}
)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --generate-line-info
                       --use_fast_math
                       >)