#if defined(ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU)
#include "OnnxOpenVinoGpuInference.h"
#include "StringUtils.h"
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>


OnnxOpenVinoGpuInference::OnnxOpenVinoGpuInference(const std::string& model_path)
{
    Ort::ThreadingOptions thread_options;
    thread_options.SetGlobalIntraOpNumThreads(1);
    Ort::SessionOptions session_options = OnnxOpenVinoGpuInference::build_session_options();

    this->env_ = std::make_unique<Ort::Env>(thread_options, ORT_LOGGING_LEVEL_ERROR, "OnnxRuntimeOpenVINOGPUInference");
    this->memory_info_ = OnnxOpenVinoGpuInference::build_memory_info();
    this->session_ = std::make_unique<Ort::Session>(*this->env_, to_wstring(model_path).c_str(), session_options);
    this->allocator_ = std::make_unique<Ort::Allocator>(*this->session_, this->memory_info_);
}


Ort::SessionOptions OnnxOpenVinoGpuInference::build_session_options()
{
    Ort::SessionOptions session_options;
    OrtOpenVINOProviderOptions open_vino_provider_options;
    open_vino_provider_options.device_type = "GPU_FP32";
    session_options.AppendExecutionProvider_OpenVINO(open_vino_provider_options);
    session_options.SetIntraOpNumThreads(0);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options.SetExecutionMode(ORT_PARALLEL);
    session_options.AddConfigEntry("session.intra_op.allow_spinning", "1");
    return session_options;
}


Ort::MemoryInfo OnnxOpenVinoGpuInference::build_memory_info()
{
    return {"OpenVINO_GPU", OrtDeviceAllocator, 0, OrtMemTypeDefault};
}

out_tensor OnnxOpenVinoGpuInference::predict(const cv::Mat& image) const
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

    cl::Buffer shared_buffer(ocl_instance->_context, CL_MEM_READ_WRITE, imgSize, NULL, &err);

    auto input_data = image_data;
    auto output_data = std::vector(output_elements, 0.0f);
    const auto input_tensor = create_tensor(this->memory_info_, input_data.data(), input_data.size(), input_shape);
    auto output_tensor = create_tensor(this->memory_info_, output_data.data(), output_data.size(), output_shape);
    this->session_->Run(Ort::RunOptions{nullptr}, &input_tensor_name_prt, &input_tensor, 1, &output_tensor_name_prt,
                        &output_tensor, 1);
    out_tensor.milliseconds = get_elapsed_time(start);
    out_tensor.predictions = output_data;
    return out_tensor;
}

out_parallel_tensors OnnxOpenVinoGpuInference::predict_all(const std::vector<cv::Mat>& images) const
{
    out_parallel_tensors parallel_tensors;
    return parallel_tensors;
}
#endif
