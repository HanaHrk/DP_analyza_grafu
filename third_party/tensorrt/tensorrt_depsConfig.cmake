cmake_minimum_required(VERSION 3.27)

message("--- CUDA Searching Start ---")
find_package(cudatoolkit_deps HINTS ${CMAKE_SOURCE_DIR}/third_party/cuda)
message("--- CUDA Searching End ---")


message("--- TensorRT Searching Start ---")


if (NOT DEFINED TensorRT_HOME)
    message(SEND_ERROR "TensorRT_HOME is required")
    return(1)
endif ()

if (NOT DEFINED OnnxRuntimeWithTensorRTProvider_HOME)
    message(SEND_ERROR "OnnxRuntimeWithTensorRTProvider_HOME is required")
    return(1)
endif ()

if (MSVC)
    FILE(GLOB OnnxRuntimeWithTensorRTProvider_LIBRARIES "${OnnxRuntimeWithTensorRTProvider_HOME}/lib/*.lib")
endif ()
set(OnnxRuntimeWithTensorRTProvider_INCLUDES "${OnnxRuntimeWithTensorRTProvider_HOME}/include")
set(OnnxRuntimeWithTensorRTProvider_BINARIES "${OnnxRuntimeWithTensorRTProvider_HOME}/bin")

add_library(tensorrt_deps INTERFACE)
target_link_libraries(tensorrt_deps INTERFACE ${TENSORRT_LIBRARIES} cudatoolkit_deps)
target_include_directories(tensorrt_deps INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include ${TENSORRT_INCLUDES})

message("--- TensorRT Searching End ---")