cmake_minimum_required(VERSION 3.22)
project(cuda_cnn CUDA)

set(CMAKE_CUDA_STANDARD 17)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

set(CUDNN_INCLUDE_DIR /nfs/volume-73-1/huangdingli/my_workspace/cuda/cudnn/include)
set(CUDNN_LIBRARY /nfs/volume-73-1/huangdingli/my_workspace/cuda/cudnn/lib)
set(CUDA_INCLUDE /usr/local/cuda-11.3/include)
set(CUDA_LIB /usr/local/cuda-11.3/lib64)

include_directories(${CUDNN_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE})
link_directories(${CUDNN_LIBRARY})
link_directories(${CUDA_LIB})

add_executable(cuda_cnn main.cu
        src/data_utils.h
        src/Layer.h
        src/utils.cuh
        src/kernels.cuh
        mnist.h
        src/Tensor.h
        src/module.h
        src/autograd.h
)

target_link_libraries(cuda_cnn cudnn cublas)