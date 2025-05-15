#pragma once
#include <fdeep/fdeep.hpp>
#include <inferencetools/InferenceEngine.hpp>


/**
 * @class FrugallyDeepEngine
 * @brief A high-performance inference engine for executing deep learning models.
 *
 * The `FrugallyDeepEngine` is a final class that combines both sequential and parallel
 * inference engine capabilities. It is designed to leverage efficient computation strategies
 * for deep learning models using the Frugally-deep library as the backend. Users can perform
 * single predictions or batch predictions while utilizing different modes of parallelization.
 *
 * This class provides an interface for loading models, performing inference on individual
 * inputs, and handling parallelized batch inference.
 */
class FrugallyDeepEngine final : public InferenceEngineSequential, public InferenceEngineParallel
{
    /**
     * @enum ParallelMode
     * @brief Enumeration defining the modes for parallel execution in the inference engine.
     *
     * Represents the options available for executing inference tasks in parallel. It provides two
     * modes:
     * - STD_THREADING: Uses standard threading for parallel processing.
     * - PARALLEL_FOREACH: Employs parallel iteration for processing tasks.
     *
     * This enumeration is used by the inference engine to determine the strategy for parallel task execution
     * when predicting outputs from input data.
     */
public:
    enum class ParallelMode
    {
        STD_THREADING,
        PARALLEL_FOREACH
    };

    /**
     * Loads a deep learning model from the specified file path into the engine.
     *
     * @param enginePath The file path to the model file to be loaded.
     */
    void loadModel(const std::string& enginePath) override;

    /**
     * @brief Executes a prediction using the input tensor and returns the resulting output tensor.
     *
     * This method takes an input tensor in the form of an `InferInput` object, processes it through
     * the loaded model, and generates an output tensor that contains the prediction results.
     *
     * The input tensor is first converted internally to a format compatible with the prediction
     * model. The output tensor produced by the model is then converted to a format suitable for the
     * application and returned as a `Tensor`.
     *
     * @param input The input tensor provided to the inference engine encapsulated in an `InferInput`
     * object. This object defines the input data and its dimensions for the prediction process.
     * @return A tensor of type `Tensor` containing the output values computed by the model during
     * the prediction process.
     */
    Tensor predict(const InferInput& input) const override;

    /**
     * Predicts outputs for all the inputs in the provided list.
     *
     * @param input A vector of InferInput objects representing the input data for predictions.
     * @return A vector of Tensor objects containing the prediction results for each input.
     */
    std::vector<Tensor> predictAll(const std::vector<InferInput>& input) const override;

    /**
     * Performs inference on all provided inputs using the specified parallelization mode.
     *
     * @param inputs A vector of InferInput objects representing the input data for inference.
     * @param mode The parallelization mode (ParallelMode) to use for inference execution.
     * @return A vector of Tensor objects containing the inference results for each input.
     */
    std::vector<Tensor> predictAll(const std::vector<InferInput>& inputs, const ParallelMode& mode) const;

    /**
     * @brief A unique pointer to an instance of fdeep::model, used for managing
     *        the lifetime and execution of a frugally deep machine learning model.
     *
     * This pointer encapsulates a frugally deep model object, which is part of the
     * FrugallyDeep library, allowing the model to be loaded, executed, and destroyed
     * in a resource-efficient manner. The use of a unique pointer ensures ownership
     * and prevents memory leaks.
     */
private:
    std::unique_ptr<fdeep::model> model_;

    /**
     * Converts an `InferInput` object into an fdeep::tensor with the correct shape and data.
     * The method ensures that the input dimensions (width, height, depth) are properly resolved
     * from either the input itself or the model's input shape configuration. If any dimension
     * cannot be determined, an exception is thrown.
     *
     * @param input An `InferInput` object containing the input data, width, height, and depth.
     *              The input data is expected to match the desired tensor format.
     * @return An `fdeep::tensor` object constructed with the resolved dimensions and input data.
     * @throw EngineRuntimeException If a required input shape (width, height, depth) is undefined.
     */
    [[nodiscard]] fdeep::tensor toFdeepTensor(const InferInput& input) const;

    /**
     * Retrieves the shape of the input data expected by the model.
     *
     * This method returns the shape of the data that the model is designed
     * to process. The shape typically describes the dimensions required for
     * the input tensor, such as the number of rows, columns, and channels
     * depending on the model architecture.
     *
     * @return An array representing the shape of the input data as integers,
     *         where each element corresponds to a dimension of the input.
     */
    ImageShape modelInputShape() const;

    /**
     * Returns a callback function for performing inference using the FrugallyDeepEngine.
     * The callback processes the given input, executes the model's prediction, and manages the
     * resulting output tensors in a thread-safe manner using atomic operations.
     *
     * @return A std::function that takes an input object of type InferInput, a vector of
     *         Tensors for output storage, and an atomic integer reference for indexing the outputs.
     */
    std::function<void(const InferInput& input, std::vector<Tensor>& output, std::atomic_int& index)> inferenceCallback() const;
};
