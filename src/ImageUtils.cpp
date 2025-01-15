//
// Created by hanah on 05.12.2024.
//

#include "ImageUtils.h"

cv::Mat load_image(const std::string& path)
{
    return imread(path, cv::IMREAD_GRAYSCALE);
}

cv::Mat modify_image(const cv::Mat& image, const int width, const int height)
{
    auto modified = cv::Mat(image);
    resize(modified, modified, cv::Size(width, height));
    modified.convertTo(modified, CV_32FC1);
    modified = 255 - modified;
    modified /= 255.0f;
    return modified;
}

std::vector<float> to_vector_input(const cv::Mat& image, const int width, const int height)
{
    std::vector<float> input;
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            input.push_back(image.at<float>(i, j));
        }
    }
    return input;
}
