#include <inferencetools/TensorRTEngineSequential.hpp>
#include <inferencetools/exception/TensorRTBuildException.hpp>
#include <inferencetools/exception/TensorRTRuntimeException.hpp>

constexpr int INPUT_BUFFER = 0;
constexpr int OUTPUT_BUFFER = 1;

void HANDLE_ERROR(const cudaError_t& e, TensorRTLogger& logger)
{
    if (e != cudaSuccess)
    {
        const auto message = "Cuda error: " + std::string(cudaGetErrorString(e));
        logger.log(nvinfer1::ILogger::Severity::kERROR, message.c_str());
        throw TensorRTRuntimeException(message);
    }
}

nvinfer1::ICudaEngine* TensorRTEngineSequential::buildTensorRTEngine(const std::string& enginePath,
                                                                     nvinfer1::ILogger& logger)
{
    const auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    constexpr auto explicitBatch = 1U << 0;
    const auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    const auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(enginePath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        throw TensorRTBuildException("Failed to parse TensorRT model (engine).");
    }
    const auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30); // 1GB WORKSPACE
    const auto engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr)
    {
        throw TensorRTBuildException("Failed to build TensorRT engine using network.");
    }
    return engine;
}

std::string TensorRTEngineSequential::getInputTensorName() const
{
    return this->context_->getEngine().getIOTensorName(0);
}

std::string TensorRTEngineSequential::getOutputTensorName() const
{
    const auto tensors = this->context_->getEngine().getNbIOTensors();
    return this->context_->getEngine().getIOTensorName(tensors - 1);
}

TensorRTEngineSequential::~TensorRTEngineSequential() = default;


void TensorRTEngineSequential::loadModel(const std::string& modelPath)
{
    constexpr auto LEVEL = nvinfer1::ILogger::Severity::kERROR;
    this->logger_ = std::make_unique<TensorRTLogger>(LEVEL);
    this->engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(buildTensorRTEngine(modelPath, *this->logger_));
    this->context_ = std::unique_ptr<nvinfer1::IExecutionContext>(this->engine_->createExecutionContext());
}

long TensorRTEngineSequential::getInputTensorSize(const InferInput& input) const
{
    // Common cases:
    // NCHW: dims = [N, C, H, W] → width is last dim
    // NHWC: dims = [N, H, W, C] → width is second to last
    // CHW: dims = [C, H, W] → width is last dim
    // HW: dims = [H, W] → width is last dim

    // Try to extract size from model
    constexpr int THREE_CHANNELS = 3;
    constexpr int ONE_CHANNEL = 1;
    std::size_t modelWidth = 0L;
    std::size_t modelHeight = 0L;
    std::size_t modelDepth = 0L;
    if (input.inputDepth != MODEL_PROPERTIES
        && input.inputHeight != MODEL_PROPERTIES
        && input.inputWidth != MODEL_PROPERTIES
    )
    {
        const auto [nbDims, d] = engine_->getTensorShape(getInputTensorName().c_str());
        if (nbDims >= 2)
        {
            // For models with explicit batch dimension (N)
            if (nbDims == 4)
            {
                // Likely NCHW or NHWC
                if (d[1] == THREE_CHANNELS || d[1] == ONE_CHANNEL)
                {
                    // NCHW format: [N, C, H, W]
                    modelDepth = d[1];
                    modelHeight = d[2];
                    modelWidth = d[3];
                }
                else
                {
                    // NHWC format: [N, H, W, C]
                    modelHeight = d[1];
                    modelWidth = d[2];
                }
            }
            else if (nbDims == 3)
            {
                // Likely CHW format
                modelHeight = d[1];
                modelWidth = d[2];
            }
            else if (nbDims == 2)
            {
                // HW format
                modelHeight = d[0];
                modelWidth = d[1];
            }
        }
    }
    const std::size_t width = input.inputWidth == MODEL_PROPERTIES ? modelWidth : input.inputWidth;
    const std::size_t height = input.inputHeight == MODEL_PROPERTIES ? modelHeight : input.inputHeight;
    const std::size_t depth = input.inputDepth == MODEL_PROPERTIES ? modelDepth : input.inputDepth;
    if (width == UNKNOWN_PROPERTY || height == UNKNOWN_PROPERTY || depth == UNKNOWN_PROPERTY)
    {
        throw TensorRTRuntimeException(
            "Input Shape is not defined properly. Model or user must define input shape for inference.");
    }


    return static_cast<long>(width * height * depth);
}

[[nodiscard]] long TensorRTEngineSequential::getOutputSize(const InferInput& input) const
{
    if (input.outputSize == MODEL_PROPERTIES)
    {
        const auto [nbDims, d] = engine_->getTensorShape(getOutputTensorName().c_str());
        long outputSize = 1;
        for (int i = 0; i < nbDims; ++i)
        {
            if (d[i] != 0)
            {
                outputSize *= static_cast<long>(d[i]);
            }
        }
        return outputSize;
    }
    return static_cast<long>(input.outputSize);
}

Tensor TensorRTEngineSequential::predict(const InferInput& input) const
{
    Tensor prediction;
    std::array<void*, 2> buffers{};

    const auto inputSize = getInputTensorSize(input);
    const auto outputSize = getOutputSize(input);
    prediction.resize(inputSize);

    HANDLE_ERROR(cudaMalloc(&buffers[INPUT_BUFFER], inputSize * sizeof(float)),
                 *this->logger_);
    HANDLE_ERROR(cudaMalloc(&buffers[OUTPUT_BUFFER], outputSize * sizeof(float)),
                 *this->logger_);
    HANDLE_ERROR(cudaMemcpy(buffers[INPUT_BUFFER], input.input.data(), inputSize * sizeof(float),
                            cudaMemcpyHostToDevice),
                 *this->logger_);
    if (!this->context_->executeV2(buffers.data()))
    {
        throw TensorRTRuntimeException("Error executing TensorRT inference.");
    }
    HANDLE_ERROR(cudaMemcpy(prediction.data(), buffers[OUTPUT_BUFFER], outputSize * sizeof(float),
                            cudaMemcpyDeviceToHost),
                 *this->logger_);
    return prediction;
}
