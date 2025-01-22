#include <future>
#if ACCELERATE_ONNX_RUNTIME_CUDA
#include "inference/OnnxRuntimeCudaInference.h"
#include "StringUtils.h"
#include "ImageUtils.h"
#include <cuda_runtime.h>
#include <memory>
#include <numeric>

OnnxRuntimeCudaInference::OnnxRuntimeCudaInference(const std::string& model_path)
{
    Ort::ThreadingOptions thread_options;
    thread_options.SetGlobalIntraOpNumThreads(1);

    this->env_ = std::make_unique<Ort::Env>(thread_options, ORT_LOGGING_LEVEL_ERROR, "OnnxRuntimeCudaInference");
    const auto session_options = build_session_options();
    this->session_ = std::make_unique<Ort::Session>(*this->env_, to_wstring(model_path).c_str(), session_options);
    this->memory_info_ = std::make_unique<Ort::MemoryInfo>("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
    this->cuda_allocator_ = std::make_unique<Ort::Allocator>(*this->session_, *this->memory_info_);
}

Ort::SessionOptions OnnxRuntimeCudaInference::build_session_options()
{
    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    cuda_options.arena_extend_strategy = 0;
    cuda_options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    cuda_options.do_copy_in_default_stream = 1;
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    session_options.SetIntraOpNumThreads(0);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options.SetExecutionMode(ORT_PARALLEL);
    session_options.AddConfigEntry("session.intra_op.allow_spinning", "1");
    return session_options;
}

OnnxRuntimeCudaInference::~OnnxRuntimeCudaInference() = default;

constexpr char kGpuGraphConfigKey[] = "gpu_graph_id";

OutTensor OnnxRuntimeCudaInference::predict(const cv::Mat& image) const
{
    OutTensor out_tensor;
    auto [start, stop] = initializeCudaEvents();

    // Get input/output tensor shapes
    auto [input_shape, output_shape] = get_input_output_shapes();
    const int input_width = static_cast<int>(input_shape[1]);
    const int input_height = static_cast<int>(input_shape[2]);

    // Calculate input/output sizes
    const auto input_size = calculate_size_from_shape(input_shape);
    const auto output_size = calculate_size_from_shape(output_shape);
    const auto input_elements = input_size / sizeof(float);
    const auto output_elements = output_size / sizeof(float);
    out_tensor.predictions.resize(output_elements);

    // Image preprocessing
    const auto image_data = process_image(image, input_width, input_height);

    // Input/output tensor names
    const auto input_tensor_name = get_input_tensor_name();
    const auto output_tensor_name = get_output_tensor_name();
    const auto input_tensor_name_prt = input_tensor_name.c_str();
    const auto output_tensor_name_prt = output_tensor_name.c_str();

    // Allocate GPU memory
    const auto input_data = std::unique_ptr<void, CudaMemoryDeleter>(
        this->cuda_allocator_->Alloc(input_size), CudaMemoryDeleter(this->cuda_allocator_.get()));
    const auto output_data = std::unique_ptr<void, CudaMemoryDeleter>(
        this->cuda_allocator_->Alloc(output_size), CudaMemoryDeleter(this->cuda_allocator_.get()));

    cudaMemcpy(input_data.get(), image_data.data(), input_size, cudaMemcpyHostToDevice);

    // Create input/output tensors
    const auto input_tensor = Ort::Value::CreateTensor<float>(*this->memory_info_,
                                                              static_cast<float*>(input_data.get()), input_elements,
                                                              input_shape.data(), input_shape.size());
    auto output_tensor = Ort::Value::CreateTensor<float>(*this->memory_info_, static_cast<float*>(output_data.get()),
                                                         output_elements, output_shape.data(), output_shape.size());

    // Run inference
    Ort::RunOptions run_options;
    run_options.AddConfigEntry(kGpuGraphConfigKey, "1");
    session_->Run(run_options, &input_tensor_name_prt, &input_tensor, 1, &output_tensor_name_prt, &output_tensor, 1);

    cudaMemcpy(out_tensor.predictions.data(), output_data.get(), output_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&out_tensor.milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return out_tensor;
}

std::vector<float> OnnxRuntimeCudaInference::process_image(const cv::Mat& image, const int width,
                                                           const int height)
{
    return to_vector_input(modify_image(image, width, height), width, height);
}

std::pair<cudaEvent_t, cudaEvent_t> OnnxRuntimeCudaInference::initializeCudaEvents()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);
    return {start, stop};
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> OnnxRuntimeCudaInference::get_input_output_shapes() const
{
    const auto input_shape = this->session_->GetInputTypeInfo(0)
                                 .GetTensorTypeAndShapeInfo()
                                 .GetShape();
    const auto output_shape = this->session_->GetOutputTypeInfo(0)
                                  .GetTensorTypeAndShapeInfo()
                                  .GetShape();
    return {input_shape, output_shape};
}

size_t OnnxRuntimeCudaInference::calculate_size_from_shape(const std::vector<int64_t>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{}) * sizeof(float);
}

OutParTensors
OnnxRuntimeCudaInference::predict_all(const std::vector<cv::Mat>& images) const
{
    OutParTensors out_par_tensors;
    return out_par_tensors;
}

void OnnxRuntimeCudaInference::async_callback(void* user_data, OrtValue** outputs, size_t num_outputs,
                                              OrtStatusPtr status)
{
}

std::string OnnxRuntimeCudaInference::get_input_tensor_name() const
{
    const Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name_prt = this->session_->GetInputNameAllocated(0, allocator);
    return input_name_prt.get();
}

std::string OnnxRuntimeCudaInference::get_output_tensor_name() const
{
    const Ort::AllocatorWithDefaultOptions allocator;
    const auto output_name_prt = this->session_->GetOutputNameAllocated(0, allocator);
    return output_name_prt.get();
}


#endif
