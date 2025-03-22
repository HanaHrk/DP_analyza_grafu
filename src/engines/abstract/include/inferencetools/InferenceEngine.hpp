#pragma once
#include <memory>
#include <utility>
#include <vector>
#include <string>

using Tensor = std::vector<float>;

struct SegmentationResult
{
    Tensor mask;

    std::vector<std::string> classLabels;
};

struct ClassificationResult
{
    int classId;

    std::string label;

    std::vector<float> confidences;

    ClassificationResult(const int classId, std::string label, const std::vector<float>& confidences)
        : classId(classId), label(std::move(label)), confidences(confidences)
    {
    }
};

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


class InferenceEngine
{
public:
    virtual ~InferenceEngine() = default;

    virtual void loadModel(const std::string& modelPath) = 0;

    virtual std::unique_ptr<SegmentationResult> segment(const Tensor& image) = 0;

    virtual std::unique_ptr<ClassificationResult> classify(const Tensor& image) = 0;

    [[nodiscard]] virtual ImageSize getSize() const = 0;

protected:
    static int getClass(const Tensor& tensor)
    {
        return static_cast<int>(std::max_element(tensor.begin(), tensor.end()) - tensor.begin());
    }
};
