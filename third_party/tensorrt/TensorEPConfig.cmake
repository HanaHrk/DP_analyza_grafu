message("--- TensorEP (OnnxRuntime Execution Provider) Searching Start ---")

# Find CUDA package in the specified third-party directory
find_package(Cuda HINTS ${CMAKE_SOURCE_DIR}/third_party/cuda REQUIRED)

# Find OnnxRuntime package in the specified third-party directory
find_package(OnnxRuntime HINTS ${CMAKE_SOURCE_DIR}/third_party/onnxruntime REQUIRED)

# Find Tensor package in the current directory
find_package(Tensor HINTS ${CMAKE_CURRENT_LIST_DIR} REQUIRED)

# Create an interface library for TensorEP
add_library(TensorEP INTERFACE)

# Link required dependencies to TensorEP
# This includes Cuda for GPU acceleration, OnnxRuntime for inference,
# and Tensor for tensor operations
target_link_libraries(TensorEP INTERFACE Cuda OnnxRuntime Tensor)

message("--- TensorEP (OnnxRuntime Execution Provider) Searching End ---")