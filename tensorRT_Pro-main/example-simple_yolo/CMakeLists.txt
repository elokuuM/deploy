cmake_minimum_required(VERSION 2.6)


project(pro)
add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

# 如果你是不同显卡，请设置为显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
set(CUDA_GEN_CODE "-gencode=arch=compute_89,code=sm_89")

# 如果你的opencv找不到，可以自己指定目录
set(OpenCV_DIR   "D:/opencv/build")

set(CUDA_DIR     "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
set(CUDNN_DIR    "D:/cudann-11.3")
set(TENSORRT_DIR "D:/TensorRT-8.6.1.6")

# set(CUDA_DIR     "/data/sxai/lean/cuda-10.2")
# set(CUDNN_DIR    "/data/sxai/lean/cudnn7.6.5.32-cuda10.2")
# set(TENSORRT_DIR "/data/sxai/lean/TensorRT-7.0.0.11")

# set(CUDA_DIR     "/data/sxai/lean/cuda-11.1")
# set(CUDNN_DIR    "/data/sxai/lean/cudnn8.2.2.26")
# set(TENSORRT_DIR "/data/sxai/lean/TensorRT-7.2.1.6")
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

find_package(CUDA REQUIRED)
find_package(OpenCV)
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_DIR}/include
    ${TENSORRT_DIR}/include
    ${CUDNN_DIR}/include
)

# 切记，protobuf的lib目录一定要比tensorRT目录前面，因为tensorRTlib下带有protobuf的so文件
# 这可能带来错误
link_directories(
    ${TENSORRT_DIR}/lib
    ${CUDA_DIR}/lib/x64
    ${CUDNN_DIR}/lib/x64
    ${OpenCV_DIR}/x64/vc16/lib
)
link_libraries(opencv_world480 opencv_world480d)
# link_libraries(${PROJECT_SOURCE_DIR}/build/Debug/cucodes)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0  -pthread -w -g")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 -O0 -Xcompiler -fPIC -g -w ${CUDA_GEN_CODE}")
file(GLOB_RECURSE cpp_srcs ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE cuda_srcs ${PROJECT_SOURCE_DIR}/src/*.cu)
cuda_add_library(cucodes STATIC ${cuda_srcs})

add_executable(pro ${cpp_srcs})

# 如果提示插件找不到，请使用dlopen(xxx.so, NOW)的方式手动加载可以解决插件找不到问题
target_link_libraries(cucodes nvinfer nvonnxparser)
target_link_libraries(cucodes cuda cublas cudart cudnn)
target_link_libraries(pro ${OpenCV_LIBS} opencv_world480 opencv_world480d)
target_link_libraries(pro cucodes)

add_custom_target(
    run
    DEPENDS pro
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/workspace
    COMMAND ./pro
)