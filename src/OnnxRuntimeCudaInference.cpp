#include <future>
#if ACCELERATE_ONNX_RUNTIME_CUDA
#include "inference/OnnxRuntimeCudaInference.h"
#include "StringUtils.h"
#include "ImageUtils.h"
#include <cuda_runtime.h>

OnnxRuntimeCudaInference::OnnxRuntimeCudaInference(const std::string &model_path) {
    Ort::ThreadingOptions thread_options;
    thread_options.SetGlobalIntraOpNumThreads(1);

    this->env_ = new Ort::Env(thread_options, ORT_LOGGING_LEVEL_ERROR, "OnnxRuntimeCudaInference");
    const auto session_options = build_session_options();
    this->session_ = new Ort::Session(*this->env_, to_wstring(model_path).c_str(), session_options);
    this->memory_info_ = new Ort::MemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
    this->cuda_allocator_ = new Ort::Allocator(*this->session_, *this->memory_info_);
}

Ort::SessionOptions OnnxRuntimeCudaInference::build_session_options() {
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

OnnxRuntimeCudaInference::~OnnxRuntimeCudaInference() {
    delete this->cuda_allocator_;
    delete this->memory_info_;
    delete this->session_;
    delete this->env_;
};

OutTensor OnnxRuntimeCudaInference::predict(const cv::Mat &image, const std::size_t out_class) const {
    OutTensor out_tensor;
    out_tensor.predictions.resize(out_class);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);
    const auto converted_image = modify_image(image);
    const auto image_data = to_vector_input(converted_image);

    const auto input_data_size = image_data.size() * sizeof(float);
    const auto output_data_size = out_class * sizeof(float);

    const char *input_names[] = {"args_0"};
    const char *output_names[] = {"dense_1"};

    const auto input_data = std::unique_ptr<void, CudaMemoryDeleter>(this->cuda_allocator_->Alloc(input_data_size),
                                                                     CudaMemoryDeleter(this->cuda_allocator_));
    const auto output_data = std::unique_ptr<void, CudaMemoryDeleter>(this->cuda_allocator_->Alloc(output_data_size),
                                                                      CudaMemoryDeleter(this->cuda_allocator_));
    cudaMemcpy(input_data.get(), image_data.data(), input_data_size, cudaMemcpyHostToDevice);

    const auto input_shape = this->session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const auto output_shape = this->session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    const auto input_tensor = Ort::Value::CreateTensor<float>(*this->memory_info_,
                                                              static_cast<float *>(input_data.get()),
                                                              image_data.size(), input_shape.data(),
                                                              input_shape.size());;
    auto output_tensor = Ort::Value::CreateTensor<float>(*this->memory_info_,
                                                         static_cast<float *>(output_data.get()),
                                                         out_tensor.predictions.size(), output_shape.data(),
                                                         output_shape.size());

    Ort::RunOptions run_options;
    run_options.AddConfigEntry("gpu_graph_id", "1");
    session_->Run(run_options, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
    cudaMemcpy(out_tensor.predictions.data(), output_data.get(), output_data_size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&out_tensor.milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return out_tensor;
}

OutParTensors
OnnxRuntimeCudaInference::predict_all(const std::vector<cv::Mat> &images, const std::size_t out_class) const {
    OutParTensors out_par_tensors;
    out_par_tensors.milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, nullptr);
    std::recursive_mutex mutex;
    std::vector<std::shared_future<std::vector<float> > > futures;
    futures.reserve(images.size());
    for (const auto &image: images) {
        std::shared_future<std::vector<float> > future = std::async(std::launch::async, [&image] {
            const auto converted_image = modify_image(image);
            const auto image_data = to_vector_input(converted_image);
            return image_data;
        }).share();
        futures.push_back(future);
    }
    for (const auto &shared_future: futures) {
        this->session_->Run
    }
    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&out_par_tensors.milliseconds, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return out_par_tensors;
}

void OnnxRuntimeCudaInference::async_callback(void *user_data, OrtValue **outputs, size_t num_outputs,
                                              OrtStatusPtr status) {
    std::cout << "Async callback" << std::endl;
    int a = 0;
}

#endif
