#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * Output Tensor.
 */
typedef struct OutTensor
{
    std::vector<float> predictions; // Probabilities of output classes.
    float milliseconds = -1; // Elapsed milliseconds of inference.
} OutTensor;

/**
 * Output Tensor for Single inference of Parallel processing.
 */
typedef struct OutParTensor
{
    std::vector<float> predictions; // Probabilities of output classes.
    float milliseconds = -1; // Elapsed milliseconds of inference.
    float offset_milliseconds = -1; // Elapsed milliseconds from the start of inference.
} OutParTensor;

/**
 * Output Tensor for All inferences of Parallel processing.
 */
typedef struct OutParTensors
{
    std::vector<OutParTensor> out_tensors; // All Output Tensors.
    float milliseconds = -1; // Elapsed milliseconds of all inferences.
} OutParTensors;

class AbstractInference
{
public:
    virtual ~AbstractInference() = default;

    /**
     * Performs inference on the given input image and returns the result as an OutTensor.
     *
     * The method is pure virtual and must be implemented by derived classes. It is responsible
     * for processing the provided image and generating predictions, including elapsed time metrics.
     *
     * @param image The input image to be inferred, represented as an OpenCV Mat object.
     * @return An OutTensor containing the predictions and inference duration in milliseconds.
     * @note The method is marked with [[nodiscard]], implying that the returned value
     *       should be used or captured to avoid potential misuse or unintended behavior.
     */
    [[nodiscard]] virtual OutTensor predict(const cv::Mat& image) const = 0;

    /**
     * Performs inference on a batch of input images and returns the results as a collection of OutParTensors.
     *
     * This method processes each image in the provided vector, generating predictions and corresponding
     * elapsed time metrics for all inferences. It is pure virtual and must be implemented by derived classes.
     *
     * @param images A vector of OpenCV Mat objects, each representing an input image for inference.
     * @return An OutParTensors struct containing the predictions, per-image inference durations,
     *         and overall elapsed time for the batch inference process.
     * @note The method is marked with [[nodiscard]], implying that the returned value
     *       should be used or captured to avoid potential misuse or unintended behavior.
     */
    [[nodiscard]] virtual OutParTensors predict_all(const std::vector<cv::Mat>& images) const =
    0;
};
