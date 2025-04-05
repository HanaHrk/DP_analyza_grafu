#pragma once
#include <string>
#include <inferencetools/InferenceEngine.hpp>

namespace sample
{
    std::string postProcessClassification(const Tensor& tensor);
}
