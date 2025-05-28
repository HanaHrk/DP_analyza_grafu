message("----- CUDA Searching Start")

# Try to find CUDA Toolkit installation
find_package(CUDAToolkit)

# If CUDA Toolkit is not found in the default locations
if (NOT CUDAToolkit_FOUND)
    # Inform user about the possibility to set CUDA globally via PATH
    message(STATUS "CudaToolkit not found implicitly. If you want to set CudaToolkit library globally, set CUDA home path to your PATH environment variable.")

    # Check if CUDAToolkit_HOME variable is defined
    if (NOT DEFINED CUDAToolkit_HOME)
        # If neither PATH nor CUDAToolkit_HOME is set, show error and exit
        message(SEND_ERROR "CudaToolkit must be defined on your PATH environment variable of CUDAToolkit_HOME must be passed to CMake.")
        return(1)
    endif ()

    # Try to find CUDA Toolkit using the specified CUDAToolkit_HOME path
    message(STATUS "Continuing searching CudaToolkit with CUDAToolkit_HOME")
    find_package(CUDAToolkit HINTS ${CUDAToolkit_HOME} REQUIRED)
endif ()

find_package(Cuda)
if (NOT Cuda_FOUND)
    # Create an interface library named 'Cuda'
    # Interface libraries are header-only libraries that just specify dependencies
    add_library(Cuda INTERFACE)

    # Link CUDA Runtime library and other CUDA libraries to our interface library
    target_link_libraries(Cuda INTERFACE CUDA::cudart ${CUDA_LIBRARIES})
endif ()

message("----- CUDA Searching End")
