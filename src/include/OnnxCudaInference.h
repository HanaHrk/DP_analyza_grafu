#pragma once
#ifdef ACCELERATE_ONNX_RUNTIME_CUDA
#include "OnnxAbstractInference.h"

class OnnxCudaInference final : public OnnxAbstractInference
{

    float handling = 0.0f;
    float gpu_time = 0.0f;

    Ort::SessionOptions build_session_options() override;

    Ort::MemoryInfo build_memory_info() override;

public:
    explicit OnnxCudaInference(const std::string& model_path);

    [[nodiscard]] out_tensor predict(const cv::Mat& image) override;

    [[nodiscard]] out_parallel_tensors predict_all(const std::vector<cv::Mat>& images) const override;
};

#endif
