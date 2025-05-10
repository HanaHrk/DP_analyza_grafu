#pragma once
#include <string>
#include <inferencetools/InferenceEngine.hpp>

namespace sample
{
    std::string getClassName(const Tensor& tensor);

    std::string postProcessClassification(const Tensor& tensor);

    void printConfusionMatrix(const std::vector<std::string>& predictedLabels,
                              const std::vector<std::string>& trueLabels);
}
