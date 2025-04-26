#pragma once
#include <stdexcept>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace sample
{
    class ImageNotFound final : public std::runtime_error
    {
    public:
        explicit ImageNotFound(const std::string& imageName) : std::runtime_error(
            "Image " + imageName + " was not found!")
        {
        }
    };

    std::vector<float> loadToVector(const cv::Mat& image, bool useBGR);

    cv::Mat preprocessImage(const cv::Mat& image, int width, int height, bool normalize);
}
