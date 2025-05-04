cmake_minimum_required(VERSION 3.27)

message("--- TensorRT Searching Start ---")

# ---------------------------------------------------------------------
# --------------------- find cuda -------------------------------------
# ---------------------------------------------------------------------
find_package(Cuda HINTS ${CMAKE_SOURCE_DIR}/third_party/cuda REQUIRED)

# ---------------------------------------------------------------------
# --------------------- find onnxruntime ------------------------------
# ---------------------------------------------------------------------
find_package(OnnxRuntime HINTS ${CMAKE_SOURCE_DIR}/third_party/onnxruntime REQUIRED)


add_library(TensorRT INTERFACE)
target_link_libraries(TensorRT INTERFACE Cuda OnnxRuntime)

message("--- TensorRT Searching End ---")