#ifdef ACCELERATE_ONNX_RUNTIME_CUDA
#include "OnnxCudaInference.h"
#include "StringUtils.h"
#include <cuda_runtime.h>
#include <fstream>

constexpr char k_gpu_graph_config_key[] = "gpu_graph_id";
std::ofstream compute_time("compute_time.log");


std::unique_ptr<void, MemoryDeleter> allocate_cuda_memory(Ort::Allocator* allocator, const size_t size)
{
    return std::unique_ptr<void, MemoryDeleter>{allocator->Alloc(size), MemoryDeleter(allocator)};
}


OnnxCudaInference::OnnxCudaInference(const std::string& model_path)
{
    Ort::ThreadingOptions thread_options;
    thread_options.SetGlobalIntraOpNumThreads(1);
    Ort::SessionOptions session_options = build_session_options();

    this->env_ = std::make_unique<Ort::Env>(thread_options, ORT_LOGGING_LEVEL_ERROR, "OnnxRuntimeCudaInference");
    this->memory_info_ = build_memory_info();
    this->session_ = std::make_unique<Ort::Session>(*this->env_, to_wstring(model_path).c_str(), session_options);
    this->allocator_ = std::make_unique<Ort::Allocator>(*this->session_, this->memory_info_);
}


Ort::SessionOptions OnnxCudaInference::build_session_options()
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


Ort::MemoryInfo OnnxCudaInference::build_memory_info()
{
    return {"Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault};
}

void copy_data_to_device(void* destination, const void* source, const size_t size)
{
    cudaMemcpy(destination, source, size, cudaMemcpyHostToDevice);
}

void copy_data_to_host(void* destination, const void* source, const size_t size)
{
    cudaMemcpy(destination, source, size, cudaMemcpyDeviceToHost);
}

out_tensor OnnxCudaInference::predict(const cv::Mat& image)
{
    out_tensor out_tensor;
    const auto start = get_time();
    auto [input_shape, output_shape] = fetch_shapes();
    const int input_width = static_cast<int>(input_shape[1]);
    const int input_height = static_cast<int>(input_shape[2]);

    const auto input_size = calculate_total_size(input_shape);
    const auto output_size = calculate_total_size(output_shape);
    const auto input_elements = input_size / sizeof(float);
    const auto output_elements = output_size / sizeof(float);
    out_tensor.predictions.resize(output_elements);

    const auto image_data = convert_image(image, input_width, input_height);

    const auto input_tensor_name = get_input_tensor_name();
    const auto output_tensor_name = get_output_tensor_name();
    const auto input_tensor_name_prt = input_tensor_name.c_str();
    const auto output_tensor_name_prt = output_tensor_name.c_str();
#ifdef PRINT_STATISTICS
    const auto handling_start = std::chrono::steady_clock::now();
#endif
    const auto input_data = allocate_cuda_memory(this->allocator_.get(), input_size);
    const auto output_data = allocate_cuda_memory(this->allocator_.get(), output_size);
    copy_data_to_device(input_data.get(), image_data.data(), input_size);
    const auto input_tensor = create_tensor(this->memory_info_, input_data.get(), input_elements, input_shape);
    auto output_tensor = create_tensor(this->memory_info_, output_data.get(), output_elements, output_shape);
#ifdef PRINT_STATISTICS
    const std::chrono::duration<float, std::milli> handling_start_millis = std::chrono::steady_clock::now() -
        handling_start;
    const auto gpu_time_start = std::chrono::steady_clock::now();
    this->handling += handling_start_millis.count();
#endif

    Ort::RunOptions run_options;
    run_options.AddConfigEntry(k_gpu_graph_config_key, "1");
    session_->Run(run_options, &input_tensor_name_prt, &input_tensor, 1, &output_tensor_name_prt, &output_tensor, 1);
#ifdef PRINT_STATISTICS
    const std::chrono::duration<float, std::milli> gpu_time_millis = std::chrono::steady_clock::now() - gpu_time_start;
    const auto handling_end = std::chrono::steady_clock::now();
    this->gpu_time += gpu_time_millis.count();
#endif
    copy_data_to_host(out_tensor.predictions.data(), output_data.get(), output_size);
#ifdef PRINT_STATISTICS
    const std::chrono::duration<float, std::milli> handling_end_millis = std::chrono::steady_clock::now() -
        handling_end;
    this->handling += handling_end_millis.count();
    compute_time << "H: " << handling << " G: " << gpu_time << std::endl;
#endif
    out_tensor.milliseconds = get_elapsed_time(start);

    return out_tensor;
}

out_parallel_tensors OnnxCudaInference::predict_all(const std::vector<cv::Mat>& images) const
{
    out_parallel_tensors tensors;
    return tensors;
}
#endif
