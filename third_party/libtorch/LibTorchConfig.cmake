
message("--- LibTorch Searching Start ---")
if (NOT DEFINED LibTorch_HOME)
    message(SEND_ERROR "LibTorch_HOME is required")
    return(1)
endif ()

find_package(Torch REQUIRED HINTS ${LibTorch_HOME})

if (NOT DEFINED LibTorch_INCLUDES)
    set(LibTorch_INCLUDES ${TORCH_INCLUDE_DIRS})
endif ()

if (NOT DEFINED LibTorch_LIBRARIES)
    set(LibTorch_LIBRARIES "${TORCH_LIBRARIES};${TORCH_CUDA_LIBRARIES}")
endif ()

if (NOT DEFINED LibTorch_BINARIES)
    set(LibTorch_BINARIES "${TORCH_INSTALL_PREFIX}/lib")
    string(REPLACE "\\" "/" LibTorch_BINARIES ${LibTorch_BINARIES})
endif ()

add_library(LibTorch INTERFACE)
target_link_libraries(LibTorch INTERFACE "${LibTorch_LIBRARIES}")
target_include_directories(LibTorch INTERFACE ${LibTorch_INCLUDES})

message("--- LibTorch Searching Start ---")
