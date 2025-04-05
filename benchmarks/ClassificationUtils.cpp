#include "ClassificationUtils.hpp"


std::string sample::postProcessClassification(const Tensor& tensor)
{
    const auto clazz = std::distance(tensor.begin(), std::max_element(tensor.begin(), tensor.end()));

    std::stringstream ss;
    ss << "Classes: " << tensor.size() << "\n";
    ss << "Predictions: ";
    for (const float value : tensor)
    {
        ss << value << ";";
    }
    ss << "\nClass: " << clazz << std::flush;
    return ss.str();
}
