#pragma once
#include <memory>
#include <utility>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

using Tensor = std::vector<float>;

enum class InferenceType : int
{
    CLASSIFICATION,
    SEGMENTATION,
    UNKNOWN,
};

typedef struct EngineInfo
{
    const std::string path;
    const InferenceType type;

    EngineInfo(std::string modelPath, const InferenceType type)
        : path(std::move(modelPath)), type(type)
    {
    }
} EngineInfo;

class ImageSize
{
public:
    const int width;
    const int height;
    const int depth;

    ImageSize(const int width, const int height, const int depth)
        : width(width), height(height), depth(depth)
    {
    }
};

struct SizedImage
{
    const Tensor& image;
    const ImageSize& size;

    SizedImage(const Tensor& image, const ImageSize& size)
        : image(image), size(size)
    {
    }
};


class InferenceEngineSequential
{
public:
    virtual ~InferenceEngineSequential() = default;

    virtual void loadModel(const EngineInfo& engineInfo) = 0;

    virtual std::unique_ptr<Tensor> predict(const cv::Mat& predictionItem,
                                            const std::function<Tensor(cv::Mat)>& transformer) const = 0;

    [[nodiscard]] virtual ImageSize getSize(const cv::Mat& def) const = 0;

    virtual InferenceType getType() const = 0;

protected:
    static int getClass(const Tensor& tensor)
    {
        return static_cast<int>(std::max_element(tensor.begin(), tensor.end()) - tensor.begin());
    }
};

class InferenceEngineParallel
{
public:
    virtual ~InferenceEngineParallel() = default;

    virtual void loadModel(const EngineInfo& modelPath) = 0;

    virtual std::vector<std::unique_ptr<Tensor>> predictAll(const std::vector<SizedImage>& images) = 0;

    [[nodiscard]] virtual ImageSize getSize() const = 0;

protected:
    static int getClass(const Tensor& tensor)
    {
        return static_cast<int>(std::max_element(tensor.begin(), tensor.end()) - tensor.begin());
    }
};
