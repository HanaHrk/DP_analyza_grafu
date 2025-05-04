
message("--- LibTorch Searching Start ---")
if (NOT DEFINED LibTorch_HOME)
    message(SEND_ERROR "LibTorch_HOME is required")
    return(1)
endif ()

find_package(Torch REQUIRED HINTS ${LibTorch_HOME})

if(MSVC)
    FILE(GLOB LibTorch_LIBRARIES "${LibTorch_HOME}/lib/*.lib")
endif ()

if (NOT DEFINED LibTorch_INCLUDES)
    set(LibTorch_INCLUDES ${TORCH_INCLUDE_DIRS})
endif ()

add_library(LibTorch INTERFACE)
target_link_libraries(LibTorch INTERFACE "${LibTorch_LIBRARIES}")
target_include_directories(LibTorch INTERFACE ${LibTorch_INCLUDES})

message("--- LibTorch Searching Start ---")
