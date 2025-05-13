#pragma once
#include <stdexcept>
#include <string>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "inferencetools/InferenceEngine.hpp"


enum class InferenceType
{
    CLASSIFICATION,
    SEGMENTATION
};

using InferPathInput = struct InferPathInput
{
    std::string path;
    InferInput input;
};

using InferPathOutput = struct InferPathOutput
{
    std::string path;
    Tensor output;
};

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

    std::vector<std::string> getAllFiles(const std::filesystem::path& rootPath);

    std::string getRandomFilePath(const std::filesystem::path& folderPath);

    std::vector<InferPathInput> getInferInputs(const std::vector<std::string>& filePaths, bool normalize, int inputWidth, int inputHeight, int inputChannels, int outputSize);
}
