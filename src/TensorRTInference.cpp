#ifdef ACCELERATE_TENSOR_RT
#include <future>
#include "ImageUtils.h"
#include "inference/TensorRTInference.h"
#include "Logger.h"
#include <Windows.h>

constexpr void HANDLE_ERROR(const cudaError_t& e)
{
    if (e != cudaSuccess)
    {
        std::cerr << "Cuda error: " << cudaGetErrorString(e) << std::endl;
        throw std::runtime_error("Cuda error:" + std::string(cudaGetErrorString(e)));
    }
}

TensorRTInference::TensorRTInference(const std::string& model_path,
                                     const nvinfer1::ILogger::Severity& severity = nvinfer1::ILogger::Severity::kINFO)
{
    this->program_logger_ = std::make_unique<ProgramLogger>(severity);
    this->engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(build_engine(model_path, *this->program_logger_));
    this->context_ = std::unique_ptr<nvinfer1::IExecutionContext>(this->engine_->createExecutionContext());
    this->init_tensor_buffers();
}

void TensorRTInference::init_tensor_buffers()
{
    const auto input_tensor_name = get_input_tensor_name();
    const auto output_tensor_name = get_output_tensor_name();
    const auto input_size = get_tensor_size(input_tensor_name);
    const auto output_size = get_tensor_size(output_tensor_name);

    HANDLE_ERROR(cudaMalloc(&this->tensor_buffers_[input_index], input_size));
    HANDLE_ERROR(cudaMalloc(&this->tensor_buffers_[output_index], output_size));
    if (!this->context_->setInputTensorAddress(input_tensor_name.c_str(), this->tensor_buffers_[input_index]))
    {
        throw std::runtime_error("Failed to set input tensor address.");
    }
    if (!this->context_->setOutputTensorAddress(output_tensor_name.c_str(), this->tensor_buffers_[output_index]))
    {
        throw std::runtime_error("Failed to set output tensor address.");
    }
}

TensorRTInference::~TensorRTInference()
{
    HANDLE_ERROR(cudaFree(this->tensor_buffers_[input_index]));
    HANDLE_ERROR(cudaFree(this->tensor_buffers_[output_index]));
}

OutTensor TensorRTInference::predict(const cv::Mat& image) const
{
    OutTensor out_tensor;
    cudaEvent_t end, start;

    // Create and record start event
    create_and_record_event(start, nullptr);

    // Access Tensors props
    const auto input_tensor_name = get_input_tensor_name();
    const auto output_tensor_name = get_output_tensor_name();
    const auto [nbDims, d] = this->context_->getEngine().getTensorShape(input_tensor_name.c_str());
    const auto input_size = get_tensor_size(input_tensor_name);
    const auto output_size = get_tensor_size(output_tensor_name);
    const auto input_width = static_cast<int>(d[1]);
    const auto input_height = static_cast<int>(d[2]);

    // Resize prediction to output size
    out_tensor.predictions.resize(output_size / sizeof(float));

    const auto image_data = process_image(image, input_width, input_height);
    HANDLE_ERROR(cudaMemcpy(this->tensor_buffers_[input_index], image_data.data(), input_size,
                            cudaMemcpyHostToDevice));
    if (!this->context_->executeV2(this->tensor_buffers_.data()))
    {
        throw std::exception("Failed to execute context");
    }

    HANDLE_ERROR(cudaMemcpy(out_tensor.predictions.data(), this->tensor_buffers_[output_index], output_size,
                            cudaMemcpyDeviceToHost));

    create_and_record_event(end, nullptr);
    out_tensor.milliseconds = compute_milliseconds(start, end);

    HANDLE_ERROR(cudaEventDestroy(end));
    HANDLE_ERROR(cudaEventDestroy(start));
    return out_tensor;
}

OutParTensors TensorRTInference::predict_all(const std::vector<cv::Mat>& images) const
{
    OutParTensors out_par_tensors;
    cudaEvent_t start_event, end_event;
    cudaStream_t stream;

    // Access Tensors props
    const auto input_tensor_name = get_input_tensor_name();
    const auto output_tensor_name = get_output_tensor_name();
    const auto [nbDims, d] = this->context_->getEngine().getTensorShape(input_tensor_name.c_str());
    const auto input_size = get_tensor_size(input_tensor_name);
    const auto output_size = get_tensor_size(output_tensor_name);
    const auto input_width = static_cast<int>(d[1]);
    const auto input_height = static_cast<int>(d[2]);

    // Initialize CUDA resources
    HANDLE_ERROR(cudaEventCreate(&start_event));
    HANDLE_ERROR(cudaEventCreate(&end_event));
    HANDLE_ERROR(cudaStreamCreate(&stream));
    HANDLE_ERROR(cudaEventRecord(start_event, stream));

    // Process each image
    for (const auto& image : images)
    {
        const auto image_data = process_image(image, input_width, input_height);
        const auto inference = process_image_inference(stream, start_event, image_data, input_size, output_size);
        out_par_tensors.out_tensors.push_back(inference);
    }

    // Final synchronization and cleanup
    HANDLE_ERROR(cudaEventRecord(end_event, stream));
    HANDLE_ERROR(cudaStreamSynchronize(stream));
    out_par_tensors.milliseconds = compute_milliseconds(start_event, end_event);

    HANDLE_ERROR(cudaStreamDestroy(stream));
    HANDLE_ERROR(cudaEventDestroy(start_event));
    HANDLE_ERROR(cudaEventDestroy(end_event));
    return out_par_tensors;
}

OutParTensor TensorRTInference::process_image_inference(const cudaStream_t& stream,
                                                        const cudaEvent_t& start_event,
                                                        const std::vector<float>& input_data,
                                                        const std::size_t input_size,
                                                        const std::size_t output_size) const
{
    cudaEvent_t image_start_event, image_end_event;
    OutParTensor tensor_result;

    // Create and record events for the image
    create_and_record_event(image_start_event, stream);

    // Copy input data and perform inference
    HANDLE_ERROR(cudaMemcpyAsync(this->tensor_buffers_[input_index], input_data.data(), input_size,
                                 cudaMemcpyHostToDevice, stream));
    if (!this->context_->enqueueV3(stream))
    {
        throw std::runtime_error("Failed to enqueue upon stream on context.");
    }

    // Retrieve output data
    tensor_result.predictions.resize(output_size / sizeof(float));
    HANDLE_ERROR(cudaMemcpyAsync(tensor_result.predictions.data(), this->tensor_buffers_[output_index], output_size,
                                 cudaMemcpyDeviceToHost, stream));

    // Synchronize and measure time
    create_and_record_event(image_end_event, stream);
    HANDLE_ERROR(cudaStreamSynchronize(stream));
    tensor_result.milliseconds = compute_milliseconds(image_start_event, image_end_event);
    tensor_result.offset_milliseconds = compute_milliseconds(start_event, image_start_event);

    // Cleanup
    HANDLE_ERROR(cudaEventDestroy(image_start_event));
    HANDLE_ERROR(cudaEventDestroy(image_end_event));

    return tensor_result;
}

void TensorRTInference::create_and_record_event(cudaEvent_t& event, const cudaStream_t& stream)
{
    HANDLE_ERROR(cudaEventCreate(&event));
    HANDLE_ERROR(cudaEventRecord(event, stream));
}

nvinfer1::ICudaEngine* TensorRTInference::build_engine(const std::string& onnx_model_path, ProgramLogger& logger)
{
    const auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger));
    constexpr auto explicitBatch = 1U << 0;
    const auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));

    const auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(onnx_model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        throw std::runtime_error("Failed to parse ONNX model.");
    }
    const auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30); // 1GB WORKSPACE
    const auto engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr)
    {
        throw std::runtime_error("Failed to build engine.");
    }
    return engine;
}

std::vector<float> TensorRTInference::process_image(const cv::Mat& image, const int width, const int height)
{
    return to_vector_input(modify_image(image, width, height), width, height);
}

float TensorRTInference::compute_milliseconds(const cudaEvent_t& start, const cudaEvent_t& end)
{
    float milliseconds = -1;
    HANDLE_ERROR(cudaEventSynchronize(end));
    HANDLE_ERROR(cudaEventElapsedTime(&milliseconds, start, end));
    return milliseconds;
}

std::string TensorRTInference::get_input_tensor_name() const
{
    return this->context_->getEngine().getIOTensorName(0);
}

std::string TensorRTInference::get_output_tensor_name() const
{
    const auto tensors = this->context_->getEngine().getNbIOTensors();
    return this->context_->getEngine().getIOTensorName(tensors - 1);
}

size_t TensorRTInference::get_tensor_size(const std::string& tensor_name) const
{
    const auto [nbDims, d] = this->context_->getEngine().getTensorShape(tensor_name.c_str());
    size_t size = 1;
    for (int i = 0; i < nbDims; i++)
    {
        size *= d[i];
    }
    return size * sizeof(float);
}

#endif
