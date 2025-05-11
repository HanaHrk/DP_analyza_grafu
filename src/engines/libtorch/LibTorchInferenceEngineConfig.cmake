# Start configuration block for LibTorch Engine
message("----- LibTorch Engine Start")

# Find and load LibTorch package from the specified location
find_package(LibTorch HINTS ${CMAKE_SOURCE_DIR}/third_party/libtorch REQUIRED)

# Create a static library for LibTorch inference engine
add_library(LibTorchInferenceEngine STATIC
        ${CMAKE_CURRENT_LIST_DIR}/LibTorchEngineSequential.cpp
)

# Set include directories for the library
target_include_directories(LibTorchInferenceEngine PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

# Link against LibTorch and AbstractInferenceEngine libraries
target_link_libraries(LibTorchInferenceEngine PUBLIC LibTorch AbstractInferenceEngine)

# For MSVC builds, copy necessary DLL files to the target directory
if (MSVC)
    copy_dlls(LibTorchInferenceEngine ${LibTorch_BINARIES})
endif ()

# End configuration block for LibTorch Engine
message("----- LibTorch Engine End")