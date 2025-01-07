#pragma once

constexpr int MODEL_INPUT_SIZE = 28;
constexpr int MODEL_OUTPUT_CLASS = 10;

#if !defined(ACCELERATE_FRUGALLY_DEEP) && !defined(ACCELERATE_TENSOR_RT) && !defined(ACCELERATE_ONNX_RUNTIME_CUDA)
    #define ACCELERATE_FRUGALLY_DEEP 1
#endif
