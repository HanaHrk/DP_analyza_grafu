#include "include/SegmentationUtils.hpp"

#include "ImageFormatException.h"

cv::Mat sample::floatArrayToImage(const std::vector<float>& floatArray, const int rows, const int cols)
{
    if (floatArray.size() != rows * cols)
    {
        throw ImageFormatException("Error: Float array size does not match rows * cols.");
    }

    cv::Mat image(rows, cols, CV_32F); // Create a single-channel float image

    // Copy the float array data to the image matrix
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            image.at<float>(i, j) = floatArray[i * cols + j];
        }
    }

    return image;
}

cv::Mat sample::postProcessSegmentation(const Tensor& tensor)
{
    const auto predictedImage = floatArrayToImage(tensor, 224, 224);
    return predictedImage * 255;
}
