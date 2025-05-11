cmake_minimum_required(VERSION 3.26)

# Define project name and language
project(InferenceBenchmark LANGUAGES CXX)

# Set C++ standard to C++17 and make it required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional features that can be enabled/disabled during configuration
option(USE_FRUGALLY_DEEP "Enable Frugally-Deep" OFF)
option(BLA "Enable Tensor" OFF)
option(USE_LIBTORCH "Enable LibTorch" OFF)

# Option to enable/disable sample benchmark execution
option(RUN_SAMPLE "Enable sample benchmark of enabled libraries" ON)

# Define the main application name
set(SAMPLE_APPLICATION InferenceBenchmark)

# List of source files for the benchmark application
set(InferenceBenchmark_SOURCES
        benchmarks/main.cpp
        benchmarks/DataLoader.cpp
        benchmarks/include/ImageFormatException.h
        benchmarks/SegmentationUtils.cpp
        benchmarks/include/SegmentationUtils.hpp
        benchmarks/ClassificationUtils.cpp
        benchmarks/ArgsParser.cpp
        benchmarks/include/ClassificationUtils.hpp
)

# Define include directories for the benchmark application
set(InferenceBenchmark_INCLUDES
        ${CMAKE_SOURCE_DIR}/benchmarks/include
)

# Create interface library for inference tools
set(LIBRARY InferenceTools)
add_library(${LIBRARY} INTERFACE)
target_include_directories(${LIBRARY} INTERFACE ${InferenceTools_INCLUDES})

# Find and link OpenCV package
find_package(CV HINTS ${CMAKE_SOURCE_DIR}/third_party/opencv REQUIRED)
target_link_libraries(${LIBRARY} INTERFACE CV)

# Find and link EngineRegistrator package
find_package(EngineRegistrator HINTS src/engines/registration REQUIRED)
target_link_libraries(${LIBRARY} INTERFACE EngineRegistrator)

# Conditionally build the sample benchmark application
if (RUN_SAMPLE)
    # Create executable from source files
    add_executable(${SAMPLE_APPLICATION} ${InferenceBenchmark_SOURCES})
    # Add include directories
    target_include_directories(${SAMPLE_APPLICATION} PRIVATE ${InferenceBenchmark_INCLUDES})
    # Link against the inference tools library
    target_link_libraries(${SAMPLE_APPLICATION} PRIVATE ${LIBRARY})
endif ()