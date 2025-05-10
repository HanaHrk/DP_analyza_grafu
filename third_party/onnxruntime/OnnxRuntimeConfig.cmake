cmake_minimum_required(VERSION 3.27)

message("--- OnnxRuntime Searching Start ---")

# ---------------------------------------------------------------------
# --------------------- validate parameters ---------------------------
# ---------------------------------------------------------------------
if (NOT DEFINED OnnxRuntime_BINARIES AND NOT DEFINED OnnxRuntime_LIBRARIES AND NOT DEFINED OnnxRuntime_INCLUDES)
    message(STATUS "OnnxRuntime_BINARIES and OnnxRuntime_LIBRARIES and OnnxRuntime_INCLUDES are not defined. Trying to search for OnnxRuntime_HOME source build.")
    if (NOT DEFINED OnnxRuntime_HOME)
        message(SEND_ERROR "OnnxRuntime_HOME is required")
        return(1)
    endif ()
endif ()

# ---------------------------------------------------------------------
# --------------------- find libraries --------------------------------
# ---------------------------------------------------------------------
if (MSVC)
    if (NOT DEFINED OnnxRuntime_INCLUDES)
        set(OnnxRuntime_INCLUDES "${OnnxRuntime_HOME}/include") # OnnxRuntime include directory (Windows)
        string(REPLACE "\\" "/" OnnxRuntime_INCLUDES ${OnnxRuntime_INCLUDES})
    endif ()
    if (NOT DEFINED OnnxRuntime_BINARIES)
        set(OnnxRuntime_BINARIES "${OnnxRuntime_HOME}/build/Windows/${CMAKE_BUILD_TYPE}/${CMAKE_BUILD_TYPE}") # OnnxRuntime binaries directory (Windows)
        string(REPLACE "\\" "/" OnnxRuntime_BINARIES ${OnnxRuntime_BINARIES})
    endif ()
    if (NOT DEFINED OnnxRuntime_LIBRARIES)
        # OnnxRuntime_LIBRARIES has same root as OnnxRuntime_BINARIES in build-from-source approach.
        file(GLOB OnnxRuntime_LIBRARIES "${OnnxRuntime_BINARIES}/*.lib") # OnnxRuntime listed libraries
    endif ()
else ()
    if (NOT DEFINED OnnxRuntime_INCLUDES)
        set(OnnxRuntime_INCLUDES "${OnnxRuntime_HOME}/include") # OnnxRuntime include directory (Linux)
    endif ()
    if (NOT DEFINED OnnxRuntime_BINARIES)
        set(OnnxRuntime_BINARIES "${OnnxRuntime_HOME}/lib") # OnnxRuntime binaries directory (Linux)
    endif ()
    if (NOT DEFINED OnnxRuntime_LIBRARIES)
        # OnnxRuntime_LIBRARIES has same root as OnnxRuntime_BINARIES in build-from-source approach.
        file(GLOB OnnxRuntime_LIBRARIES "${OnnxRuntime_BINARIES}/*.so") # OnnxRuntime listed libraries (linux)
    endif ()
endif ()
message(STATUS "OnnxRuntime_HOME=${OnnxRuntime_HOME}")
message(STATUS "OnnxRuntime_INCLUDES=${OnnxRuntime_INCLUDES}")
message(STATUS "OnnxRuntime_BINARIES=${OnnxRuntime_BINARIES}")
message(STATUS "OnnxRuntime_LIBRARIES=${OnnxRuntime_LIBRARIES}")

# ---------------------------------------------------------------------
# --------------------- define library --------------------------------
# ---------------------------------------------------------------------
add_library(OnnxRuntime INTERFACE)

# ---------------------------------------------------------------------
# --------------------- link libraries --------------------------------
# ---------------------------------------------------------------------
target_link_libraries(OnnxRuntime INTERFACE ${OnnxRuntime_LIBRARIES})
target_include_directories(OnnxRuntime INTERFACE ${OnnxRuntime_INCLUDES})
message("--- OnnxRuntime Searching End ---")