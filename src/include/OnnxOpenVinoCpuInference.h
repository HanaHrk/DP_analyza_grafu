#pragma once
#ifdef ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU
#include "OnnxAbstractInference.h"

class OnnxOpenVinoCpuInference final : public OnnxAbstractInference
{
protected:
    Ort::SessionOptions build_session_options() override;

    Ort::MemoryInfo build_memory_info() override;

public:
    explicit OnnxOpenVinoCpuInference(const std::string& model_path);

    [[nodiscard]] out_tensor predict(const cv::Mat& image) const override;

    [[nodiscard]] out_parallel_tensors predict_all(const std::vector<cv::Mat>& images) const override;
};

#endif
