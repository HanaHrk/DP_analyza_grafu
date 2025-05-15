#pragma once
#include <inferencetools/InferenceEngine.hpp>
#include <opencv2/opencv.hpp>

namespace sample
{
    /**
     * Converts a 1D float array into a 2D single-channel OpenCV matrix (cv::Mat) representation.
     *
     * @param floatArray The input vector of floats representing the image data.
     * @param rows The number of rows in the output image matrix.
     * @param cols The number of columns in the output image matrix.
     * @return A cv::Mat object representing the image, constructed from the input float array.
     * @throws ImageFormatException If the size of the float array does not match rows * cols.
     */
    cv::Mat floatArrayToImage(const Tensor& floatArray, int rows, int cols);

    /**
     * Processes the given segmentation tensor and converts it to an image.
     * The input tensor is expected to contain segmentation data to be processed
     * into a single-channel image and scaled for further visualization or usage.
     *
     * @param tensor The input Tensor containing segmentation data to be processed.
     *               Should represent a flattened 2D data structure.
     * @return A cv::Mat object representing the processed segmentation image.
     *         The image will be scaled and returned in a single-channel format.
     * @throws ImageFormatException if the size of the tensor does not match the expected dimensions.
     */
    cv::Mat postProcessSegmentation(const Tensor& tensor);
}
