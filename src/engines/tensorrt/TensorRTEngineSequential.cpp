#include <locale>
#include <string>
#include <codecvt>
#include <numeric>
#include <inferencetools/TensorRTEngineSequential.hpp>

#include <inferencetools/exception/TensorRTRuntimeException.hpp>
#include <inferencetools/exception/TensorRTBuildException.hpp>
#include <cuda_runtime.h>

constexpr int INPUT_BUFFER = 0;
constexpr int OUTPUT_BUFFER = 1;

std::wstring toWString(const std::string& s)
{
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(s);
}

constexpr void HANDLE_CUDA(const cudaError_t& error)
{
    if (error != cudaSuccess)
    {
        throw TensorRTRuntimeException("CUDA error: " + std::to_string(error));
    }
}

Ort::SessionOptions TensorRTEngineSequential::createSessionOptions()
{
    Ort::SessionOptions sessionOptions;
    OrtCUDAProviderOptions cudaProviderOptions;
    cudaProviderOptions.device_id = 0;
    OrtTensorRTProviderOptions tensorRTProviderOptions;
    tensorRTProviderOptions.device_id = 0;
    tensorRTProviderOptions.has_user_compute_stream = false;
    tensorRTProviderOptions.trt_max_partition_iterations = 128;
    tensorRTProviderOptions.trt_min_subgraph_size = 1;
    tensorRTProviderOptions.trt_max_workspace_size = 1073741824;
    tensorRTProviderOptions.trt_fp16_enable = true;
    tensorRTProviderOptions.trt_int8_enable = false;
    tensorRTProviderOptions.trt_int8_calibration_table_name = nullptr;
    tensorRTProviderOptions.trt_int8_use_native_calibration_table = false;
    tensorRTProviderOptions.trt_dla_enable = false;
    tensorRTProviderOptions.trt_dla_core = 0;
    tensorRTProviderOptions.trt_dump_subgraphs = 0;
    tensorRTProviderOptions.trt_engine_cache_enable = false;
    tensorRTProviderOptions.trt_engine_cache_path = nullptr;
    tensorRTProviderOptions.trt_engine_decryption_enable = false;
    tensorRTProviderOptions.trt_engine_decryption_lib_path = nullptr;
    tensorRTProviderOptions.trt_force_sequential_engine_build = false;
    try
    {
        sessionOptions.AppendExecutionProvider_TensorRT(tensorRTProviderOptions);
        //sessionOptions.AppendExecutionProvider_CUDA(cudaProviderOptions);
    }
    catch (const Ort::Exception& e)
    {
        throw TensorRTBuildException("Failed to append Cuda and TensorRT provider: " + std::string(e.what()));
    }
    return sessionOptions;
}

Ort::MemoryInfo TensorRTEngineSequential::createMemoryInfo()
{
    return {"Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault};
}

Ort::ThreadingOptions TensorRTEngineSequential::createThreadingOptions()
{
    Ort::ThreadingOptions threadOptions;
    threadOptions.SetGlobalIntraOpNumThreads(1);
    return threadOptions;
}

void TensorRTEngineSequential::loadModel(const std::string& modelPath)
{
    const auto threadOptions = createThreadingOptions();
    this->env_ = std::make_unique<Ort::Env>(threadOptions, ORT_LOGGING_LEVEL_ERROR, "OnnxRuntimeTensorRTEngine");
    const auto sessionOptions = createSessionOptions();
    this->memoryInfo_ = createMemoryInfo();
#if _MSC_VER
    this->session_ = std::make_unique<Ort::Session>(*this->env_, toWString(modelPath).c_str(), sessionOptions);
#else
    this->session_ = std::make_unique<Ort::Session>(*this->env_, modelPath.c_str(), sessionOptions);
#endif
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

    void *inputBuffer, *outputBuffer = nullptr;
    HANDLE_CUDA(cudaMalloc(&inputBuffer, inputSize * sizeof(float)));
    HANDLE_CUDA(cudaMemcpy(inputBuffer, input.input.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_CUDA(cudaMalloc(&outputBuffer, outputSize * sizeof(float)));
    const auto inputTensor = Ort::Value::CreateTensor<float>(this->memoryInfo_, static_cast<float*>(inputBuffer),
                                                             inputSize, inputShape.data(), inputShape.size());
    auto outputTensor = Ort::Value::CreateTensor<float>(this->memoryInfo_, static_cast<float*>(outputBuffer),
                                                        outputSize, outputShape.data(), outputShape.size());

    session_->Run(Ort::RunOptions{nullptr}, &inputTensorNamePtr, &inputTensor, 1, &outputTensorNamePtr, &outputTensor,
                  1);
    HANDLE_CUDA(cudaMemcpy(outTensor.data(), outputBuffer, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    HANDLE_CUDA(cudaFree(inputBuffer));
    HANDLE_CUDA(cudaFree(outputBuffer));
    return outTensor;
}

uint64_t TensorRTEngineSequential::getTensorSize(const TensorShape& tensorShape)
{
    return std::abs(std::accumulate(tensorShape.begin(), tensorShape.end(), 1, std::multiplies<>()));
}

std::vector<int64_t> absVector(std::vector<int64_t>& v)
{
    std::transform(v.begin(), v.end(), v.begin(), [](const int n)
    {
        return std::abs(n);
    });
    return v;
}

TensorShape TensorRTEngineSequential::getInputTensorShape(const InferInput& input) const
{
    if (input.inputDepth == MODEL_PROPERTIES ||
        input.inputWidth == MODEL_PROPERTIES ||
        input.inputHeight == MODEL_PROPERTIES)
    {
        auto outputShape = this->session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.empty())
        {
            throw TensorRTRuntimeException("Input Shape is not defined properly. Model or user must define input shape for inference.");
        }
        return absVector(outputShape);
    }
    throw TensorRTRuntimeException("Input shape must be defined inside model.");
}

TensorShape TensorRTEngineSequential::getOutputTensorShape(const InferInput& input) const
{
    if (input.outputSize == MODEL_PROPERTIES)
    {
        auto outputShape = this->session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (outputShape.empty())
        {
            throw TensorRTRuntimeException("Output Shape is not defined properly. Model must define output shape for inference.");
        }
        return absVector(outputShape);
    }
    throw TensorRTRuntimeException("Output shape must be defined inside model.");
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
