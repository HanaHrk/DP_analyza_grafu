#pragma once
#if defined(ACCELERATE_ONNX_RUNTIME_CUDA)
#include "OnnxAbstractInference.h"

class OnnxCudaInference final : public OnnxAbstractInference
{
    Ort::SessionOptions build_session_options() override;

    Ort::MemoryInfo build_memory_info() override;

public:
    explicit OnnxCudaInference(const std::string& model_path);

    [[nodiscard]] out_tensor predict(const cv::Mat& image) const override;

    [[nodiscard]] out_parallel_tensors predict_all(const std::vector<cv::Mat>& images) const override;
};

#endif
