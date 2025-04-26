
message("--- LibTorch Searching Start ---")
if (NOT DEFINED LibTorch_HOME)
    message(SEND_ERROR "LibTorch_HOME is required")
    return(1)
endif ()

find_package(Torch REQUIRED HINTS ${LibTorch_HOME})

if(MSVC)
    FILE(GLOB TORCH_LIBRARIES "${LibTorch_HOME}/lib/*.lib")
endif ()

add_library(libtorch_deps INTERFACE)
target_link_libraries(libtorch_deps INTERFACE "${TORCH_LIBRARIES}")
target_include_directories(libtorch_deps INTERFACE ${TORCH_INCLUDE_DIRS})

message("--- LibTorch Searching Start ---")
