#include "ClassificationUtils.hpp"

std::string sample::getClassName(const Tensor& tensor)
{
    const int clazz = std::distance(tensor.begin(), std::max_element(tensor.begin(), tensor.end()));
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
    return CLASSES[clazz];
}

std::string sample::postProcessClassification(const Tensor& tensor)
{
    std::stringstream ss;
    ss << "Classes: " << tensor.size() << "\n";
    ss << "Predictions: ";
    for (const float value : tensor)
    {
        ss << value << ";";
    }
    ss << "\nClass: " << getClassName(tensor) << std::flush;
    return ss.str();
}

void sample::printConfusionMatrix(const std::vector<std::string>& predictedLabels,
                          const std::vector<std::string>& trueLabels)
{
    // Create a map to store unique class labels
    std::map<std::string, size_t> label_to_index;

    // Get unique labels
    for (const auto& label : trueLabels)
    {
        if (label_to_index.find(label) == label_to_index.end())
        {
            label_to_index[label] = label_to_index.size();
        }
    }

    // Initialize confusion matrix with zeros
    const size_t num_classes = label_to_index.size();
    std::vector confusion_matrix(num_classes, std::vector(num_classes, 0));

    // Fill the confusion matrix
    for (size_t i = 0; i < predictedLabels.size(); ++i)
    {
        const size_t pred_idx = label_to_index[predictedLabels[i]];
        const size_t true_idx = label_to_index[trueLabels[i]];
        confusion_matrix[true_idx][pred_idx]++;
    }

    // Calculate per-class totals
    std::vector class_totals(num_classes, 0);
    for (size_t i = 0; i < num_classes; ++i)
    {
        for (size_t j = 0; j < num_classes; ++j)
        {
            class_totals[i] += confusion_matrix[i][j];
        }
    }

    // Print header
    std::cout << "\nConfusion Matrix (percentages per class):\n";
    std::cout << std::setw(15) << "Predicted >";

    // Print predicted labels
    for (const auto& [fst, snd] : label_to_index)
    {
        std::cout << std::setw(10) << fst;
    }
    std::cout << "\nActual v\n";

    // Print matrix content with row labels and percentages
    for (const auto& [fst, snd] : label_to_index)
    {
        std::cout << std::setw(15) << fst;
        for (size_t j = 0; j < num_classes; ++j)
        {
            float percentage = 0.0f;
            if (class_totals[snd] > 0)
            {
                percentage = static_cast<float>(confusion_matrix[snd][j]) /
                    static_cast<float>(class_totals[snd]) * 100.0f;
            }
            std::cout << std::setw(9) << std::fixed << std::setprecision(1) << percentage << "%";
        }
        // Print total samples for this class
        std::cout << " (" << class_totals[snd] << " samples)";
        std::cout << '\n';
    }

    // Calculate and print overall accuracy
    int correct = 0;
    int total = 0;
    for (size_t i = 0; i < num_classes; ++i)
    {
        correct += confusion_matrix[i][i];
        total += class_totals[i];
    }

    const float accuracy = static_cast<float>(correct) / static_cast<float>(total) * 100.0f;
    std::cout << "\nOverall Accuracy: " << std::fixed << std::setprecision(1)
        << accuracy << "%";

    // Print per-class accuracies
    std::cout << "\n\nPer-class Accuracies:\n";
    for (const auto& [fst, snd] : label_to_index)
    {
        const size_t idx = snd;
        float class_accuracy = 0.0f;
        if (class_totals[idx] > 0)
        {
            class_accuracy = static_cast<float>(confusion_matrix[idx][idx]) / static_cast<float>(class_totals[idx]) *
                100.0f;
        }
        std::cout << std::setw(15) << fst << ": "
            << std::fixed << std::setprecision(1) << class_accuracy << "%\n";
    }
    std::cout << std::endl;
}
