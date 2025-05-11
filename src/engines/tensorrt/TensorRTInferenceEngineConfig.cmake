# Begin TensorRT Engine configuration
message("----- TensorRT Engine Start")

# Find TensorRT package from the specified location
find_package(TensorEP HINTS "${CMAKE_SOURCE_DIR}/third_party/tensorrt" REQUIRED)

# Define static library for TensorRT inference engine
add_library(TensorRTInferenceEngine STATIC
        ${CMAKE_CURRENT_LIST_DIR}/TensorRTEngineSequential.cpp
        ${CMAKE_CURRENT_LIST_DIR}/include/inferencetools/exception/TensorRTBuildException.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/inferencetools/exception/TensorRTRuntimeException.hpp
)

# Set include directories for the TensorRT engine
target_include_directories(TensorRTInferenceEngine PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

# Link required libraries to TensorRT engine
target_link_libraries(TensorRTInferenceEngine PUBLIC TensorEP AbstractInferenceEngine)

# Windows-specific configuration
if (MSVC)
    # Copy required DLLs for Windows builds
    copy_dlls(TensorRTInferenceEngine ${OnnxRuntime_BINARIES})
    copy_dlls(TensorRTInferenceEngine ${Tensor_BINARIES})
endif ()

# End TensorRT Engine configuration
message("----- TensorRT Engine End")