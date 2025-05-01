#pragma once
#include <memory>
#include <utility>
#include <vector>
#include <string>

#include <opencv2/opencv.hpp>

using Tensor = std::vector<float>;

constexpr std::size_t MODEL_PROPERTIES = 0;
constexpr std::size_t UNKNOWN_PROPERTY = 0;

typedef struct ImageSize
{
    const std::size_t width;
    const std::size_t height;
    const std::size_t depth;

    ImageSize(const std::size_t width, const std::size_t height, const std::size_t depth) : width(width),
        height(height), depth(depth)
    {
    }
} ImageShape;

typedef std::vector<int64_t> TensorShape;

typedef struct InferInput
{
    const Tensor input;
    const std::size_t inputWidth;
    const std::size_t inputHeight;
    const std::size_t inputDepth;
    const std::size_t outputSize;

    InferInput(Tensor input, const std::size_t inputWidth, const std::size_t inputHeight,
               const std::size_t inputDepth, const std::size_t outputSize) : input(std::move(input)),
                                                                             inputWidth(inputWidth),
                                                                             inputHeight(inputHeight),
                                                                             inputDepth(inputDepth),
                                                                             outputSize(outputSize)
    {
    }

    explicit InferInput(Tensor input) : input(std::move(input)),
                                        inputWidth(MODEL_PROPERTIES),
                                        inputHeight(MODEL_PROPERTIES),
                                        inputDepth(MODEL_PROPERTIES),
                                        outputSize(MODEL_PROPERTIES)

    {
    }
} InferInput;

class InferenceEngineSequential
{
public:
    virtual ~InferenceEngineSequential() = default;

    virtual void loadModel(const std::string& enginePath) = 0;

    [[nodiscard]] virtual Tensor predict(const InferInput& input) const = 0;
};
