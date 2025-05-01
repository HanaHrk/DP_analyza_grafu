#include <locale>
#include <string>
#include <codecvt>
#include <numeric>
#include <inferencetools/TensorRTEngineSequential.hpp>
#include <inferencetools/exception/TensorRTRuntimeException.hpp>

constexpr int INPUT_BUFFER = 0;
constexpr int OUTPUT_BUFFER = 1;

std::wstring toWString(const std::string& s)
{
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(s);
}


Ort::SessionOptions TensorRTEngineSequential::createSessionOptions()
{
    Ort::SessionOptions sessionOptions;
    const auto tensorStatus = OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0);
    const auto cudaStatus = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

    std::cout << OrtGetApiBase()->GetApi(ORT_API_VERSION)->GetErrorMessage(tensorStatus) << ", " << OrtGetApiBase()->GetApi(ORT_API_VERSION)->GetErrorMessage(cudaStatus) << std::endl;
    return sessionOptions;
}

Ort::MemoryInfo TensorRTEngineSequential::createMemoryInfo()
{
    return {"TensorRT", OrtArenaAllocator, 0, OrtMemTypeDefault};
}

Ort::ThreadingOptions TensorRTEngineSequential::createThreadingOptions()
{
    Ort::ThreadingOptions threadOptions;
    threadOptions.SetGlobalIntraOpNumThreads(1);
    return threadOptions;
}

void TensorRTEngineSequential::loadModel(const std::string& modelPath)
{
    const auto sessionOptions = createSessionOptions();
    const auto threadOptions = createThreadingOptions();
    this->env_ = std::make_unique<Ort::Env>(threadOptions, ORT_LOGGING_LEVEL_INFO, "OnnxRuntimeTensorRTEngine");
    this->memoryInfo_ = createMemoryInfo();
    this->session_ = std::make_unique<Ort::Session>(*this->env_, toWString(modelPath).c_str(), sessionOptions);
}

Tensor TensorRTEngineSequential::predict(const InferInput& input) const
{
    Tensor outTensor;
    const auto inputShape = getInputTensorShape(input);
    const auto outputShape = getOutputTensorShape(input);
    const auto inputSize = getTensorSize(inputShape);
    const auto outputSize = getTensorSize(outputShape);

    const auto inputTensorName = getInputTensorName();
    const auto outputTensorName = getOutputTensorName();
    const auto inputTensorNamePtr = inputTensorName.c_str();
    const auto outputTensorNamePtr = outputTensorName.c_str();

    outTensor.resize(outputSize);
    const auto inputTensor = Ort::Value::CreateTensor<float>(this->memoryInfo_, const_cast<float*>(input.input.data()), inputSize, inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(this->memoryInfo_, outTensor.data(), outputSize, outputShape.data(), outputShape.size());

    const Ort::RunOptions run_options;
    session_->Run(run_options, &inputTensorNamePtr, &inputTensor, 1, &outputTensorNamePtr, &outputTensor, 1);

    return outTensor;
}

uint64_t TensorRTEngineSequential::getTensorSize(const TensorShape& tensorShape)
{
    return std::accumulate(tensorShape.begin(), tensorShape.end(), 1, std::multiplies<>());
}

TensorShape TensorRTEngineSequential::getInputTensorShape(const InferInput& input) const
{
    if (input.inputDepth == MODEL_PROPERTIES || input.inputWidth == MODEL_PROPERTIES || input.inputHeight == MODEL_PROPERTIES)
    {
        const auto outputShape = this->session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.empty())
        {
            throw TensorRTRuntimeException("Output Shape is not defined properly. Model or user must define input shape for inference.");
        }
        return outputShape;
    }
    return TensorShape{1, static_cast<int64_t>(input.inputWidth), static_cast<int64_t>(input.inputHeight), static_cast<int64_t>(input.inputDepth)};
}

TensorShape TensorRTEngineSequential::getOutputTensorShape(const InferInput& input) const
{
    if (input.outputSize == MODEL_PROPERTIES)
    {
        const auto outputShape = this->session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.empty())
        {
            throw TensorRTRuntimeException("Output Shape is not defined properly. Model or user must define input shape for inference.");
        }
        return outputShape;
    }
    return TensorShape{1, static_cast<int64_t>(input.outputSize)};
}

std::string TensorRTEngineSequential::getInputTensorName() const
{
    const Ort::AllocatorWithDefaultOptions allocator;
    const auto inputNamePtr = this->session_->GetInputNameAllocated(0, allocator);
    return inputNamePtr.get();
}

std::string TensorRTEngineSequential::getOutputTensorName() const
{
    const Ort::AllocatorWithDefaultOptions allocator;
    const auto outputNamePrt = this->session_->GetOutputNameAllocated(0, allocator);
    return outputNamePrt.get();
}
