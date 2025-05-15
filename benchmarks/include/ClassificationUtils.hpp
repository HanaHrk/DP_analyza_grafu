#pragma once
#include <string>
#include <map>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <inferencetools/InferenceEngine.hpp>

namespace sample
{
    /**
     * Determines the class name associated with the index of the maximum value in the input tensor.
     * The tensor represents a probability distribution over predefined classes.
     * The list of predefined classes includes:
     * - 0: "area"
     * - 1: "heatmap"
     * - 2: "horizontal_bar"
     * - 3: "horizontal_interval"
     * - 4: "line"
     * - 5: "manhattan"
     * - 6: "map"
     * - 7: "pie"
     * - 8: "scatter"
     * - 9: "scatter-line"
     * - 10: "surface"
     * - 11: "venn"
     * - 12: "vertical_bar"
     * - 13: "vertical_box"
     * - 14: "vertical_interval"
     *
     * @param tensor The tensor containing the probabilities for each class.
     * Each element in the tensor represents the likelihood of the corresponding class.
     * @return The name of the class corresponding to the highest probability in the tensor.
     */
    std::string getClassName(const Tensor& tensor);

    /**
     * Processes the given classification tensor and generates a formatted string
     * containing information about the classification output.
     * The string includes the total number of classes, prediction values, and
     * the determined class name.
     *
     * @param tensor The input Tensor containing classification data. Each element
     *               in the tensor represents a prediction score for a specific class.
     * @return A formatted string representation of the classification results. The string
     *         includes the number of classes, prediction values, and the name of the predicted class.
     */
    std::string postProcessClassification(const Tensor& tensor);

    /**
     * Prints the confusion matrix and classification results metrics, including
     * per-class accuracy and overall accuracy, for a classification task.
     *
     * The confusion matrix compares the predicted labels against the true labels,
     * calculating occurrences for each true-predicted label pair. Percentages
     * are computed relative to the total samples for each class. Overall accuracy
     * and per-class accuracies are also calculated and displayed.
     *
     * @param predictedLabels A vector of strings representing the predicted labels
     *                        from the classifier for each sample.
     * @param trueLabels      A vector of strings representing the ground truth
     *                        labels for each sample in the dataset.
     */
    void printConfusionMatrix(const std::vector<std::string>& predictedLabels,
                              const std::vector<std::string>& trueLabels);
}
