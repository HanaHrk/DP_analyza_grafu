message("----- OnnxRuntime Searching Start")

# Check if essential OnnxRuntime paths are defined
# If not, try to use OnnxRuntime_HOME as fallback
if (NOT DEFINED OnnxRuntime_BINARIES AND NOT DEFINED OnnxRuntime_LIBRARIES AND NOT DEFINED OnnxRuntime_INCLUDES)
    message(STATUS "OnnxRuntime_BINARIES and OnnxRuntime_LIBRARIES and OnnxRuntime_INCLUDES are not defined. Trying to search for OnnxRuntime_HOME source build.")
    # Verify OnnxRuntime_HOME is defined, exit if not
    if (NOT DEFINED OnnxRuntime_HOME)
        message(SEND_ERROR "OnnxRuntime_HOME is required")
        return(1)
    endif ()
endif ()

# Platform-specific path configuration
if (MSVC)
    # Windows-specific configurations
    if (NOT DEFINED OnnxRuntime_INCLUDES)
        set(OnnxRuntime_INCLUDES "${OnnxRuntime_HOME}/include") # Set include path for Windows
        string(REPLACE "\\" "/" OnnxRuntime_INCLUDES ${OnnxRuntime_INCLUDES}) # Normalize path separators
    endif ()
    if (NOT DEFINED OnnxRuntime_BINARIES)
        # Set binaries path for Windows, using build type-specific directory
        set(OnnxRuntime_BINARIES "${OnnxRuntime_HOME}/build/Windows/${CMAKE_BUILD_TYPE}/${CMAKE_BUILD_TYPE}")
        string(REPLACE "\\" "/" OnnxRuntime_BINARIES ${OnnxRuntime_BINARIES}) # Normalize path separators
    endif ()
    if (NOT DEFINED OnnxRuntime_LIBRARIES)
        # Find all .lib files in the binaries directory
        file(GLOB OnnxRuntime_LIBRARIES "${OnnxRuntime_BINARIES}/*.lib")
    endif ()
else ()
    # Linux-specific configurations
    if (NOT DEFINED OnnxRuntime_INCLUDES)
        set(OnnxRuntime_INCLUDES "${OnnxRuntime_HOME}/include") # Set include path for Linux
    endif ()
    if (NOT DEFINED OnnxRuntime_BINARIES)
        set(OnnxRuntime_BINARIES "${OnnxRuntime_HOME}/lib") # Set binaries path for Linux
    endif ()
    if (NOT DEFINED OnnxRuntime_LIBRARIES)
        # Find all .so files in the binaries directory
        file(GLOB OnnxRuntime_LIBRARIES "${OnnxRuntime_BINARIES}/*.so")
    endif ()
endif ()

# Log the configured paths for debugging
message(STATUS "OnnxRuntime_HOME=${OnnxRuntime_HOME}")
message(STATUS "OnnxRuntime_INCLUDES=${OnnxRuntime_INCLUDES}")
message(STATUS "OnnxRuntime_BINARIES=${OnnxRuntime_BINARIES}")
message(STATUS "OnnxRuntime_LIBRARIES=${OnnxRuntime_LIBRARIES}")

# Create an interface library target for OnnxRuntime
add_library(OnnxRuntime INTERFACE)

# Configure the interface library with include directories and link libraries
target_link_libraries(OnnxRuntime INTERFACE ${OnnxRuntime_LIBRARIES})
target_include_directories(OnnxRuntime INTERFACE ${OnnxRuntime_INCLUDES})

message("----- OnnxRuntime Searching End")