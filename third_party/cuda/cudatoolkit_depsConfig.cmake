cmake_minimum_required(VERSION 3.27)

find_package(CUDAToolkit)

if (NOT CUDAToolkit_FOUND)
    message(STATUS "CudaToolkit not found implicitly. If you want to set CudaToolkit library globally, set path to your PATH environment variable.")
    if (NOT DEFINED CUDAToolkit_HOME)
        message(SEND_ERROR "CudaToolkit must be defined on your PATH environment variable of CUDAToolkit_HOME must be passed to CMake.")
        return(1)
    endif ()
    message(STATUS "Continuing searching CudaToolkit with CUDAToolkit_HOME")
endif ()

add_library(cudatoolkit_deps INTERFACE)
target_link_libraries(cudatoolkit_deps INTERFACE CUDA::cudart)
install(TARGETS cudatoolkit_deps DESTINATION "${CUDAToolkit_HOME}/bin")