set(CMAKE_BUILD_TYPE Release)
cmake_minimum_required(VERSION 3.24)

project(yolov5)

#set(CMAKE_BUILD_TYPE Release)
#添加opencv静态库
set(OpenCV_DIR D:/opencv/build) # xxxx目录包含OpenCVConfig.cmake
find_package(OpenCV REQUIRED core highgui imgproc)	# 找到opencv库
include_directories(${OpenCV_INCLUDE_DIRS})


find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#添加ncnn
include_directories(D:/ncnn/protobuf_build/install/include/ncnn)
set(ncnn_DIR "D:/ncnn/protobuf_build/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)

add_executable(${PROJECT_NAME} yolov5_infer.cpp)
target_link_libraries(${PROJECT_NAME} ncnn ${OpenCV_LIBRARIES})  #D:/opencv/build/x64/vc16/lib/opencv_world480.lib

