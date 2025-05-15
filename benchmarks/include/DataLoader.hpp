#pragma once
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "inferencetools/InferenceEngine.hpp"

/**
 * @brief Types of inference operations supported by the system
 */
enum class InferenceType
{
    CLASSIFICATION,  ///< Image classification inference type
    SEGMENTATION    ///< Image segmentation inference type
};

/**
 * @brief Structure linking a file path with its corresponding inference input
 */
struct InferPathInput
{
    std::string path;   ///< Path to the input file
    InferInput input;   ///< Preprocessed inference input data
};

/**
 * @brief Structure linking a file path with its corresponding inference output
 */
struct InferPathOutput
{
    std::string path;   ///< Path to the input file
    Tensor output;      ///< Inference output tensor
};

namespace sample
{
    /**
     * @brief Exception thrown when an image file cannot be found
     */
    class ImageNotFound final : public std::runtime_error
    {
    public:
        /**
         * @brief Constructs an ImageNotFound exception
         * @param imageName Name of the image that was not found
         */
        explicit ImageNotFound(const std::string& imageName);
    };

    /**
     * @brief Converts an OpenCV image matrix to a vector of float values
     * @param image Input image as OpenCV Mat
     * @param useBGR Flag to determine if BGR color order should be used
     * @return Vector of float values representing the image
     */
    std::vector<float> loadToVector(const cv::Mat& image, bool useBGR);

    /**
     * @brief Preprocesses an image for inference
     * @param image Input image as OpenCV Mat
     * @param width Target width for resizing
     * @param height Target height for resizing
     * @param normalize Flag indicating whether to normalize pixel values
     * @return Preprocessed image as OpenCV Mat
     */
    cv::Mat preprocessImage(const cv::Mat& image, int width, int height, bool normalize);

    /**
     * @brief Retrieves all files from a specified directory
     * @param rootPath Path to the root directory to search
     * @return Vector of strings containing file paths
     * @throws std::filesystem::filesystem_error if directory access fails
     */
    std::vector<std::string> getAllFiles(const std::filesystem::path& rootPath);

    /**
     * @brief Gets a random file path from the specified folder
     * @param folderPath Path to the folder to choose from
     * @return String containing a random file path
     * @throws std::filesystem::filesystem_error if directory access fails
     */
    std::string getRandomFilePath(const std::filesystem::path& folderPath);

    /**
     * @brief Prepares inference inputs for a batch of files
     * @param filePaths Vector of paths to input files
     * @param normalize Whether to normalize the input data
     * @param inputWidth Width of the input images
     * @param inputHeight Height of the input images
     * @param inputChannels Number of input channels
     * @param outputSize Size of the output
     * @return Vector of InferPathInput structures
     * @throws ImageNotFound if any input file cannot be found
     */
    std::vector<InferPathInput> getInferInputs(const std::vector<std::string>& filePaths, 
                                              bool normalize, 
                                              int inputWidth, 
                                              int inputHeight, 
                                              int inputChannels, 
                                              int outputSize);
}