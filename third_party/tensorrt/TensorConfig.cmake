message("----- Tensor Searching Start")

# Check if the essential Tensor-related paths are defined
# These include binary files, libraries, and include directories
if (NOT DEFINED Tensor_BINARIES AND NOT DEFINED Tensor_LIBRARIES AND NOT DEFINED Tensor_INCLUDES)
    message(STATUS "Tensor_BINARIES and Tensor_LIBRARIES and Tensor_INCLUDES are not defined. Trying to search for Tensor_HOME source build.")
    # If paths are not defined, check for Tensor_HOME environment variable
    if (NOT DEFINED Tensor_HOME)
        message(SEND_ERROR "Tensor_HOME is required")
        return(1)
    endif ()
endif ()

# Set the include directory path if not already defined
# Uses Tensor_HOME as the base directory
if (NOT DEFINED Tensor_INCLUDES)
    set(Tensor_INCLUDES "${Tensor_HOME}/include")
    # Convert Windows path separators to forward slashes for CMake compatibility
    string(REPLACE "\\" "/" Tensor_INCLUDES ${Tensor_INCLUDES})
endif ()

# Set the binary directory path if not already defined
if (NOT DEFINED Tensor_BINARIES)
    set(Tensor_BINARIES "${Tensor_HOME}/lib")
    # Convert Windows path separators to forward slashes for CMake compatibility
    string(REPLACE "\\" "/" Tensor_BINARIES ${Tensor_BINARIES})
endif ()

# Set the library paths if not already defined
if (NOT DEFINED Tensor_LIBRARIES)
    if (MSVC)
        # For Microsoft Visual Studio Compiler, look for .lib files
        file(GLOB Tensor_LIBRARIES "${Tensor_BINARIES}/*.lib")
    else ()
        # For other compilers (e.g., GCC, Clang), look for .so files
        file(GLOB Tensor_LIBRARIES "${Tensor_BINARIES}/*.so")
    endif ()
endif ()

# Create an interface library target named 'Tensor'
add_library(Tensor INTERFACE)
# Add include directories to the interface library
target_include_directories(Tensor INTERFACE ${Tensor_INCLUDES})
# Link the required libraries to the interface library
target_link_libraries(Tensor INTERFACE ${Tensor_LIBRARIES})

message("----- Tensor Searching End")
