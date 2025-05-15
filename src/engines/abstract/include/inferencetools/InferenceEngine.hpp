#pragma once
#include <utility>
#include <vector>
#include <string>

using Tensor = std::vector<float>;

/**
 * @brief Specifies the default value for model-related properties.
 *
 * This constant represents the initial or undefined state for properties
 * such as input dimensions (width, height, depth) and output size
 * in the inference process. It is used as a placeholder value in scenarios
 * where these parameters are dynamically determined or not explicitly set.
 *
 * Functions or methods referencing this value typically replace it with
 * concrete dimensions based on the model's characteristics or input data
 * during the inference pipeline execution.
 */
constexpr std::size_t MODEL_PROPERTIES = 0;

/**
 * @var UNKNOWN_PROPERTY
 * @brief Represents an unspecified property or attribute with an undefined purpose.
 *
 * The UNKNOWN_PROPERTY variable serves as a placeholder or representation of
 * a property whose functionality, type, or intended usage is currently unclear
 * or to be determined. Its purpose may vary depending on the specific context
 * or requirement where it is used.
 */
constexpr std::size_t UNKNOWN_PROPERTY = 0;

/**
 * @struct ImageSize
 * @brief Represents the dimensions of an image, including width, height, and depth.
 *
 * The ImageSize structure provides a convenient way to define and manage the
 * size parameters of an image. It stores the width, height, and depth as
 * constant values and initializes them through its constructor.
 */
typedef struct ImageSize
{
    /**
     * @var width
     * @brief Represents the width dimension of an image or tensor.
     *
     * This variable specifies the width of the image or data structure,
     * typically used in various computations related to image or tensor
     * dimensions in an inference engine or graphics-related processing.
     */
    const std::size_t width;

    /**
     * @var height
     * @brief Represents the height dimension of an image or tensor.
     *
     * The height variable is a constant size measurement that defines
     * the vertical resolution or extent of an image, tensor, or data representation
     * in computational contexts. It is typically used in conjunction with
     * width and depth to fully describe an image or 3D shape.
     */
    const std::size_t height;

    /**
     * @var depth
     * @brief Represents the depth or level of a specific entity or structure.
     *
     * This variable is typically used to indicate the vertical or hierarchical distance,
     * layer, or intensity in a given context. The exact meaning of depth
     * depends on the specific application or implementation.
     */
    const std::size_t depth;


    /**
     * @brief Constructs an ImageSize object with the specified dimensions.
     *
     * This constructor initializes the width, height, and depth to represent
     * the dimensions of an image or 3D tensor.
     *
     * @param width The width of the image or 3D tensor.
     * @param height The height of the image or 3D tensor.
     * @param depth The depth (e.g., number of channels) of the image or 3D tensor.
     */
    ImageSize(const std::size_t width, const std::size_t height, const std::size_t depth) : width(width),
                                                                                            height(height), depth(depth)
    {
    }
} ImageShape;

typedef std::vector<int64_t> TensorShape;


/**
 * @class InferInput
 * @brief Represents an input tensor to be passed to an inference engine.
 *
 * The InferInput class encapsulates the properties and behavior of an input
 * tensor, including its name, shape, data type, and associated data. It is
 * designed to provide necessary details for preparing and feeding input
 * data into an inference model.
 */
typedef struct InferInput
{
    /**
     * @brief Represents the input tensor provided to the inference engine.
     *
     * This variable contains the data that will be processed by the model
     * and serves as the starting point for inference operations.
     **/
    const Tensor input;

    /**
     * @brief Represents the width of the input tensor or data provided to the system.
     *
     * This variable defines the horizontal dimension of the input, typically used
     * in processing or resizing operations within a computational pipeline.
     **/
    const std::size_t inputWidth;

    /**
     * @brief Represents the height dimension of the input data.
     *
     * This variable defines the vertical size of the input tensor or image provided for processing.
     **/
    const std::size_t inputHeight;

    /**
     * @brief Represents the depth of the input tensor for the model.
     *
     * This variable defines the number of channels or depth information
     * in the input data that the model processes.
     **/
    const std::size_t inputDepth;

    /**
     * @brief Specifies the size of the output tensor produced by the inference engine.
     *
     * This variable defines the total number of elements expected in the output tensor
     * after the inference process is complete. It is typically determined by the model
     * architecture and configuration.
     *
     * This value is used as a reference for managing and validating the output tensor
     * data, ensuring its dimensions align with the expected results of the inference.
     */
    const std::size_t outputSize;

    /**
     * @brief Constructs an instance of InferInput with the specified tensor and input dimensions.
     *
     * This constructor initializes an InferInput object used for inference, where the input tensor
     * data and its dimensional specifications are provided. It defines the spatial and depth
     * properties of the input tensor as well as the expected output size for the inference pipeline.
     *
     * @param input A tensor containing the input data for the inference process.
     * @param inputWidth The width of the input tensor.
     * @param inputHeight The height of the input tensor.
     * @param inputDepth The depth (channels) of the input tensor.
     * @param outputSize The expected size of the output produced by the inference process.
     */
    InferInput(Tensor input, const std::size_t inputWidth, const std::size_t inputHeight,
               const std::size_t inputDepth, const std::size_t outputSize) : input(std::move(input)),
                                                                             inputWidth(inputWidth),
                                                                             inputHeight(inputHeight),
                                                                             inputDepth(inputDepth),
                                                                             outputSize(outputSize)
    {
    }

    /**
     * @brief Constructs an InferInput object with a tensor input, initializing other properties to default values.
     *
     * This constructor takes a tensor object and sets the input tensor for the InferInput instance.
     * The dimensions and properties of the input (such as width, height, depth, and output size)
     * are initialized to the default value defined by MODEL_PROPERTIES.
     *
     * @param input A tensor representing the input data, which is moved into the InferInput instance.
     */
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

    /**
     * @brief Loads a pre-trained model from the given file path into the engine.
     *
     * This function is used to initialize and prepare the model specified at the provided
     * file location so that it can be utilized within the inference engine. The implementation
     * details of this process may vary based on specific requirements and configurations.
     *
     * @param enginePath The path to the model file to be loaded. The file must exist, and the format
     *                 should be compatible with the engine's expectations.
     * @exception Derived implementations may handle errors such as file access issues,
     *            unsupported or corrupted model formats, or initialization failures.
     */
    virtual void loadModel(const std::string& enginePath) = 0;

    /**
     * @brief Performs inference on the provided input data and returns the predicted output tensor.
     *
     * This function is responsible for processing the input data through the inference engine
     * and generating a prediction output. The input must contain properly formatted dimensions
     * and data in accordance with the model requirements.
     *
     * @param input A constant reference to an `InferInput` structure containing the input tensor
     *              and its associated metadata, such as dimensions and expected output size.
     *              The input must adhere to the pre-defined model properties.
     *
     * @return A `Tensor` representing the output of the inference. The output tensor contains
     *         the predicted values generated by the model after processing the input.
     *
     * @exception Derived implementations may handle inference failures caused by invalid input,
     *            runtime issues during inference, or incompatible model configurations.
     */
    [[nodiscard]] virtual Tensor predict(const InferInput& input) const = 0;

};

class InferenceEngineParallel
{
public:
    virtual ~InferenceEngineParallel() = default;

    /**
     * @brief Loads a model into the inference engine from the specified file path.
     *
     * This method is responsible for loading the pre-trained model into the inference engine
     * using the file located at the given path. The exact behavior of this function, including
     * how the model is loaded and validated, is defined in the derived implementation.
     *
     * @param enginePath The file path to the serialized model that needs to be loaded into the engine.
     *                   This path must point to a valid model file that is compatible with the engine.
     * @exception Derived implementations may define specific behaviors and raise exceptions, such as
     *            file-not-found errors, invalid model format, or deserialization issues.
     */
    virtual void loadModel(const std::string& enginePath) = 0;

    /**
     * @brief Performs parallel prediction for all the provided input data.
     *
     * This method takes a vector of input data structures (`InferInput`) and performs inference
     * on all of them to produce the corresponding output tensors. Each `InferInput` represents a set of
     * input features and associated properties for the inference engine. The function is expected to be
     * implemented by a derived class, where the specific inference logic is defined.
     *
     * @param input A vector of `InferInput` objects containing the input data, dimensions, and expected output size
     *              for each inference task.
     * @return A vector of output tensors (`Tensor`), where each tensor corresponds to the inference result of each
     *         respective input in the provided input vector. The output tensors are ordered to match the input order.
     * @note The method is marked [[nodiscard]], indicating that the caller must use the returned result.
     * @exception Specific derived implementations may define the behavior and specify potential exceptions, including
     *            situations such as invalid input data or errors during inference execution.
     */
    [[nodiscard]] virtual std::vector<Tensor> predictAll(const std::vector<InferInput>& input) const = 0;
};
