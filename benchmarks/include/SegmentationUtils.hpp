#pragma once
#include <inferencetools/InferenceEngine.hpp>
#include <opencv2/opencv.hpp>

namespace sample
{
    cv::Mat floatArrayToImage(const Tensor& floatArray, int rows, int cols);

    cv::Mat postProcessSegmentation(const Tensor& tensor);
}
