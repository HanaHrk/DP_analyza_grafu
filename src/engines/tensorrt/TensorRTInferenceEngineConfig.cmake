cmake_minimum_required(VERSION 3.10)

message("--- TensorRT Engine Start ---")

# Add TensorRT dependencies
find_package(TensorEP HINTS "${CMAKE_SOURCE_DIR}/third_party/tensorrt" REQUIRED)


# Create engine library
add_library(TensorRTInferenceEngine STATIC
        ${CMAKE_CURRENT_LIST_DIR}/TensorRTEngineSequential.cpp
        ${CMAKE_CURRENT_LIST_DIR}/include/inferencetools/exception/TensorRTBuildException.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/inferencetools/exception/TensorRTRuntimeException.hpp
)
target_include_directories(TensorRTInferenceEngine PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(TensorRTInferenceEngine PUBLIC TensorEP AbstractInferenceEngine)

if (MSVC)
    copy_dlls(TensorRTInferenceEngine ${OnnxRuntime_BINARIES})
    copy_dlls(TensorRTInferenceEngine ${Tensor_BINARIES})
endif ()

message("--- TensorRT Engine End ---")