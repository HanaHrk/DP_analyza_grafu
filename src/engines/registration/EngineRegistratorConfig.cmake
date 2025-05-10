cmake_minimum_required(VERSION 3.10)

message("--- Engine Registrator Start ---")

# Create engine library
add_library(EngineRegistrator STATIC
        ${CMAKE_CURRENT_LIST_DIR}/EngineFactory.cpp
)

# Add Abstract Engine dependencies
find_package(AbstractInferenceEngine HINTS ${CMAKE_SOURCE_DIR}/src/engines/abstract REQUIRED)

# Add subdirectories conditionally
if (USE_FRUGALLY_DEEP)
    # Add frugally_deep engine
    find_package(FrugallyDeepInferenceEngine HINTS ${CMAKE_SOURCE_DIR}/src/engines/frugally_deep REQUIRED)
    list(APPEND COMPILE_OPTIONS -DUSE_FRUGALLY_DEEP=1)
    list(APPEND ENGINE_LIBRARIES FrugallyDeepInferenceEngine)
endif ()

if (BLA)
    # Add tensor_rt engine
    find_package(TensorRTInferenceEngine HINTS ${CMAKE_SOURCE_DIR}/src/engines/tensorrt REQUIRED)
    list(APPEND COMPILE_OPTIONS -DBLA=1)
    list(APPEND ENGINE_LIBRARIES TensorRTInferenceEngine)
endif ()

if (USE_LIBTORCH)
    # Add tensor_rt engine
    find_package(LibTorchInferenceEngine HINTS ${CMAKE_SOURCE_DIR}/src/engines/libtorch REQUIRED)
    list(APPEND COMPILE_OPTIONS -DUSE_LIBTORCH=1)
    list(APPEND ENGINE_LIBRARIES LibTorchInferenceEngine)
endif ()

target_compile_definitions(EngineRegistrator PUBLIC ${COMPILE_OPTIONS})
target_include_directories(EngineRegistrator PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(EngineRegistrator PUBLIC CV ${ENGINE_LIBRARIES} AbstractInferenceEngine)

message("--- Engine Registrator End ---")