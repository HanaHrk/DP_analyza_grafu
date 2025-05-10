cmake_minimum_required(VERSION 3.27)

message("--- Tensor Searching Start ---")


# ---------------------------------------------------------------------
# --------------------- validate parameters ---------------------------
# ---------------------------------------------------------------------
if (NOT DEFINED Tensor_BINARIES AND NOT DEFINED Tensor_LIBRARIES AND NOT DEFINED Tensor_INCLUDES)
    message(STATUS "Tensor_BINARIES and Tensor_LIBRARIES and Tensor_INCLUDES are not defined. Trying to search for Tensor_HOME source build.")
    if (NOT DEFINED Tensor_HOME)
        message(SEND_ERROR "Tensor_HOME is required")
        return(1)
    endif ()
endif ()

    if (NOT DEFINED Tensor_INCLUDES)
        set(Tensor_INCLUDES "${Tensor_HOME}/include") # Tensor include directory
        string(REPLACE "\\" "/" Tensor_INCLUDES ${Tensor_INCLUDES})
    endif ()
    if (NOT DEFINED Tensor_BINARIES)
        set(Tensor_BINARIES "${Tensor_HOME}/lib")
        string(REPLACE "\\" "/" Tensor_BINARIES ${Tensor_BINARIES})
    endif ()
    if (NOT DEFINED Tensor_LIBRARIES)
        # Tensor_LIBRARIES has same root as Tensor_BINARIES in build-from-source approach.
        if(MSVC)
            file(GLOB Tensor_LIBRARIES "${Tensor_BINARIES}/*.lib") # Tensor listed libraries
        else ()
            file(GLOB Tensor_LIBRARIES "${Tensor_BINARIES}/*.so") # Tensor listed libraries
        endif ()
    endif ()

add_library(Tensor INTERFACE)
target_include_directories(Tensor INTERFACE ${Tensor_INCLUDES})
target_link_libraries(Tensor INTERFACE ${Tensor_LIBRARIES})


message("--- Tensor Searching End ---")