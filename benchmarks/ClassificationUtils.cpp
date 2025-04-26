#include "ClassificationUtils.hpp"


std::string sample::postProcessClassification(const Tensor& tensor)
{
    std::map<int, std::string> CLASSES = {
        {0, "area"},
        {1, "heatmap"},
        {2, "horizontal_bar"},
        {3, "horizontal_interval"},
        {4, "line"},
        {5, "manhattan"},
        {6, "map"},
        {7, "pie"},
        {8, "scatter"},
        {9, "scatter-line"},
        {10, "surface"},
        {11, "venn"},
        {12, "vertical_bar"},
        {13, "vertical_box"},
        {14, "vertical_interval"}
    };
    const int clazz = std::distance(tensor.begin(), std::max_element(tensor.begin(), tensor.end()));

    std::stringstream ss;
    ss << "Classes: " << tensor.size() << "\n";
    ss << "Predictions: ";
    for (const float value : tensor)
    {
        ss << value << ";";
    }
    ss << "\nClass: " << CLASSES[clazz] << std::flush;
    return ss.str();
}
