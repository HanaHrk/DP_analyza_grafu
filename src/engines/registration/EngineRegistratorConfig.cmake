message("----- Engine Registrator Start")

# Define a static library target named EngineRegistrator
add_library(EngineRegistrator STATIC
        ${CMAKE_CURRENT_LIST_DIR}/EngineFactory.cpp
)

# Find and include the required AbstractInferenceEngine package
find_package(AbstractInferenceEngine HINTS ${CMAKE_SOURCE_DIR}/src/engines/abstract REQUIRED)

# Conditional build configuration for Frugally Deep engine
if (USE_FRUGALLY_DEEP)
    # Find and include FrugallyDeepInferenceEngine package
    find_package(FrugallyDeepInferenceEngine HINTS ${CMAKE_SOURCE_DIR}/src/engines/frugally_deep REQUIRED)
    # Add compiler definition to enable Frugally Deep
    list(APPEND COMPILE_OPTIONS -DUSE_FRUGALLY_DEEP=1)
    # Add FrugallyDeepInferenceEngine to the list of engine libraries
    list(APPEND ENGINE_LIBRARIES FrugallyDeepInferenceEngine)
endif ()

# Conditional build configuration for TensorRT engine
if (USE_TENSOR_RT)
    # Find and include TensorRTInferenceEngine package
    find_package(TensorRTInferenceEngine HINTS ${CMAKE_SOURCE_DIR}/src/engines/tensorrt REQUIRED)
    # Add compiler definition to enable TensorRT
    list(APPEND COMPILE_OPTIONS -DBLA=1)
    # Add TensorRTInferenceEngine to the list of engine libraries
    list(APPEND ENGINE_LIBRARIES TensorRTInferenceEngine)
endif ()

# Conditional build configuration for LibTorch engine
if (USE_LIBTORCH)
    # Find and include LibTorchInferenceEngine package
    find_package(LibTorchInferenceEngine HINTS ${CMAKE_SOURCE_DIR}/src/engines/libtorch REQUIRED)
    # Add compiler definition to enable LibTorch
    list(APPEND COMPILE_OPTIONS -DUSE_LIBTORCH=1)
    # Add LibTorchInferenceEngine to the list of engine libraries
    list(APPEND ENGINE_LIBRARIES LibTorchInferenceEngine)
endif ()

# Configure the EngineRegistrator target
# Set compile definitions
target_compile_definitions(EngineRegistrator PUBLIC ${COMPILE_OPTIONS})
# Add include directories
target_include_directories(EngineRegistrator PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
# Link required libraries
target_link_libraries(EngineRegistrator PUBLIC ${ENGINE_LIBRARIES} AbstractInferenceEngine)

# Print end message for Engine Registrator configuration
message("----- Engine Registrator End")