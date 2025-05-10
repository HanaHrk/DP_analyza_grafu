cmake_minimum_required(VERSION 3.27)

message("--- TensorEP (OnnxRuntime Execution Provider) Searching Start ---")

# ---------------------------------------------------------------------
# --------------------- Find Cuda -------------------------------------
# ---------------------------------------------------------------------
find_package(Cuda HINTS ${CMAKE_SOURCE_DIR}/third_party/cuda REQUIRED)

# ---------------------------------------------------------------------
# --------------------- Find OnnxRuntime ------------------------------
# ---------------------------------------------------------------------
find_package(OnnxRuntime HINTS ${CMAKE_SOURCE_DIR}/third_party/onnxruntime REQUIRED)

# ---------------------------------------------------------------------
# --------------------- Find TensorEP ---------------------------------
# ---------------------------------------------------------------------
find_package(Tensor HINTS ${CMAKE_CURRENT_LIST_DIR} REQUIRED)

add_library(TensorEP INTERFACE)
target_link_libraries(TensorEP INTERFACE Cuda OnnxRuntime Tensor)

message("--- TensorEP (OnnxRuntime Execution Provider) Searching End ---")