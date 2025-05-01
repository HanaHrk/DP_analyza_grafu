cmake_minimum_required(VERSION 3.10)

message("--- TensorRT Engine Start ---")

# Add TensorRT dependencies
find_package(tensorrt_deps HINTS "${CMAKE_SOURCE_DIR}/third_party/tensorrt" REQUIRED)

# Create engine library
add_library(tensorrt_engine STATIC
        ${CMAKE_CURRENT_LIST_DIR}/TensorRTEngineSequential.cpp
        ${CMAKE_CURRENT_LIST_DIR}/include/inferencetools/exception/TensorRTBuildException.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/inferencetools/exception/TensorRTRuntimeException.hpp
)
target_include_directories(tensorrt_engine PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_link_libraries(tensorrt_engine PUBLIC tensorrt_deps abstract_engine)

if (MSVC)
    # copy_dlls(tensorrt_engine ${TENSORRT_BINARIES})
endif ()

message("--- TensorRT Engine End ---")