#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

typedef struct OutTensor
{
    std::vector<float> predictions;
    float milliseconds = -1;
} out_tensor;

typedef struct OutParTensor
{
    std::vector<float> predictions;
    float milliseconds = -1;
    float offset_milliseconds = -1;
} OutParTensor;

typedef struct OutParTensors
{
    std::vector<OutParTensor> out_tensors;
    float milliseconds = -1;
} out_parallel_tensors;

class AbstractInference
{
protected:
    static std::vector<float> convert_image(const cv::Mat& image, uint32_t width, uint32_t height);

    static std::chrono::time_point<std::chrono::steady_clock> get_time();

    static float get_elapsed_time(const std::chrono::time_point<std::chrono::steady_clock>& start);

public:
    virtual ~AbstractInference() = default;

    [[nodiscard]] virtual out_tensor predict(const cv::Mat& image) const = 0;

    [[nodiscard]] virtual out_parallel_tensors predict_all(const std::vector<cv::Mat>& images) const = 0;
};
