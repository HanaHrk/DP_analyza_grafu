cmake_minimum_required(VERSION 3.30.5)
project(InferenceBenchmark LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options to enable/disable libraries
option(USE_FRUGALLY_DEEP "Enable Frugally-Deep" OFF)
option(USE_TENSORRT "Enable TensorRT" OFF)
option(USE_LIBTORCH "Enable LibTorch" OFF)

# Option to build samples benchmarks
option(RUN_SAMPLE "Enable sample benchmark of enabled libraries" ON)

# Add executable configuration
set(SAMPLE_APPLICATION InferenceBenchmark)
set(InferenceBenchmark_SOURCES
        benchmarks/main.cpp
        benchmarks/DataLoader.cpp
        benchmarks/include/ImageFormatException.h
        benchmarks/SegmentationUtils.cpp
        benchmarks/include/SegmentationUtils.hpp
        benchmarks/ClassificationUtils.cpp
        benchmarks/include/ClassificationUtils.hpp
)
set(InferenceBenchmark_INCLUDES
        ${CMAKE_SOURCE_DIR}/benchmarks/include
)


# Add library configuration
set(LIBRARY InferenceTools)
# Create InferenceTools library
add_library(${LIBRARY} INTERFACE)
target_include_directories(${LIBRARY} INTERFACE ${InferenceTools_INCLUDES})

find_package(CV HINTS ${CMAKE_SOURCE_DIR}/third_party/opencv REQUIRED)
target_link_libraries(${LIBRARY} INTERFACE CV)
find_package(EngineRegistrator HINTS src/engines/registration REQUIRED)
target_link_libraries(${LIBRARY} INTERFACE EngineRegistrator)


# Create sample executable
if (RUN_SAMPLE)
    add_executable(${SAMPLE_APPLICATION} ${InferenceBenchmark_SOURCES})
    target_include_directories(${SAMPLE_APPLICATION} PRIVATE ${InferenceBenchmark_INCLUDES})
    target_link_libraries(${SAMPLE_APPLICATION} PRIVATE ${LIBRARY})
endif ()
