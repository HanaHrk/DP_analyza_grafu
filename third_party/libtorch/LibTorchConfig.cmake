message("----- LibTorch Searching Start")

# Check if LibTorch home directory is defined
if (NOT DEFINED LibTorch_HOME)
    message(SEND_ERROR "LibTorch_HOME is required")
    return(1)
endif ()

# Find CUDA package in the specified location
find_package(Cuda HINTS ${CMAKE_SOURCE_DIR}/third_party/cuda REQUIRED)

# Display LibTorch home directory for debugging
message(STATUS "LibTorch_HOME=${LibTorch_HOME}")

# Set Torch directory to LibTorch home if defined
if (DEFINED LibTorch_HOME)
    set(Torch_DIR ${LibTorch_HOME})
endif ()

# Find and load the Torch package
find_package(Torch CONFIG REQUIRED)

# Set LibTorch include directories if not already defined
if (NOT DEFINED LibTorch_INCLUDES)
    set(LibTorch_INCLUDES ${TORCH_INCLUDE_DIRS})
endif ()

# Set LibTorch libraries if not already defined
# Combines both Torch and Torch CUDA libraries
if (NOT DEFINED LibTorch_LIBRARIES)
    set(LibTorch_LIBRARIES "${TORCH_LIBRARIES};${TORCH_CUDA_LIBRARIES}")
endif ()

# Set LibTorch binary directory if not already defined
# Converts Windows path separators to forward slashes
if (NOT DEFINED LibTorch_BINARIES)
    set(LibTorch_BINARIES "${TORCH_INSTALL_PREFIX}/lib")
    string(REPLACE "\\" "/" LibTorch_BINARIES ${LibTorch_BINARIES})
endif ()

# Create an interface library for LibTorch
add_library(LibTorch INTERFACE)

# Link libraries and include directories to the interface library
target_link_libraries(LibTorch INTERFACE "${LibTorch_LIBRARIES}")
target_include_directories(LibTorch INTERFACE ${LibTorch_INCLUDES})

message("----- LibTorch Searching Start")