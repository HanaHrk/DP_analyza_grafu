cmake_minimum_required(VERSION 3.27)


find_package(CUDAToolkit)
if (NOT CUDAToolkit_FOUND)
    message(STATUS "CudaToolkit not found implicitly. If you want to set CudaToolkit library globally, set CUDA home path to your PATH environment variable.")
    if (NOT DEFINED CUDAToolkit_HOME)
        message(SEND_ERROR "CudaToolkit must be defined on your PATH environment variable of CUDAToolkit_HOME must be passed to CMake.")
        return(1)
    endif ()
    message(STATUS "Continuing searching CudaToolkit with CUDAToolkit_HOME")
    find_package(CUDAToolkit HINTS ${CUDAToolkit_HOME} REQUIRED)
endif ()

enable_language(CUDA)
add_library(Cuda INTERFACE)
target_link_libraries(Cuda INTERFACE CUDA::cudart ${CUDA_LIBRARIES})