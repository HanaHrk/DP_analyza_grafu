#pragma once
#include <torch/script.h>
#include <inferencetools/InferenceEngine.hpp>


/**
 * @brief Represents an implementation of the `InferenceEngineSequential` using LibTorch.
 *
 * This class provides functionality to load a pre-trained model and perform inference
 * using the PyTorch JIT scripting module. It is designed to extend the `InferenceEngineSequential`
 * with specific capabilities related to the LibTorch library.
 *
 * The class is marked `final`, which prevents further inheritance and ensures a fixed behavior
 * for this implementation. The loaded model is stored as a private object of type
 * `torch::jit::script::Module`.
 */
class LibtorchEngineSequential final : public InferenceEngineSequential
{
    /**
     * @brief Destructor for the LibtorchEngineSequential class.
     *
     * This destructor is responsible for releasing resources allocated by an instance
     * of the `LibtorchEngineSequential` class. It is declared as `override` to
     * ensure that the proper cleanup logic is executed for the derived class
     * when the base class pointer is used to delete the object instance.
     *
     * The `default` specifier indicates that the compiler will generate the default
     * implementation for the destructor.
     */
public:
    ~LibtorchEngineSequential() override = default;

    /**
     * @brief Loads a pre-trained LibTorch model from the specified file path.
     *
     * This method initializes and prepares the LibTorch model based on the
     * given file path, ensuring it is ready for inference. The model is set to
     * evaluation mode after loading. If an error occurs during the process,
     * a LibTorchBuildException is thrown.
     *
     * @param enginePath The file path to the LibTorch model to be loaded.
     *                   The file must exist and be a valid, supported LibTorch
     *                   model format.
     * @exception LibTorchBuildException Thrown in case of loading errors, such as
     *            missing file, corrupted model, or incompatibility issues.
     */
    void loadModel(const std::string& enginePath) override;

    /**
     * @brief Predicts the output tensor using the loaded LibTorch model given the input tensor.
     *
     * This method processes the provided input tensor through the loaded LibTorch model
     * and generates an output tensor based on the model's inference. The input tensor
     * is converted to the appropriate format for the model and the prediction result
     * is returned. Any runtime errors during inference are caught and handled.
     *
     * @param input The input data for the model, encapsulated within an InferInput structure.
     *              It contains the tensor data as well as its dimensions and other properties.
     * @return The predicted output tensor as a vector of floating-point values.
     *         This represents the result of the inference process executed by the model.
     * @throws LibTorchRuntimeException if any error occurs during the prediction process.
     */
    [[nodiscard]] Tensor predict(const InferInput& input) const override;

    /**
     * @brief A smart pointer to hold the TorchScript module for the neural network model.
     *
     * This member variable represents a `std::unique_ptr` to manage the lifecycle
     * of the TorchScript `Module` object. It is used to handle and perform inference
     * operations on a pre-loaded PyTorch serialized model (TorchScript format). The model
     * is loaded using the `loadModel` method and subsequently utilized in
     * the `predict` method for performing inference.
     *
     * The `unique_ptr` ensures that ownership of the model object is bound
     * to the `LibtorchEngineSequential` instance, and the memory is released
     * when the instance is destroyed. The model is initialized once during
     * the loading phase and remains constant for the lifetime of the engine instance.
     *
     * The model operates in inference mode (`eval`) after it is loaded.
     */
private:
    std::unique_ptr<torch::jit::script::Module> model_;
};
