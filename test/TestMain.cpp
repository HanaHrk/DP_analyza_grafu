#include <ImageUtils.h>
#include <fstream>
#include "AbstractInference.h"
#include "inference/FrugallyDeepInference.h"
#include "inference/TensorRTInference.h"
#include "TestUtils.h"
#include "MathUtils.h"
#include "inference/OnnxRuntimeCudaInference.h"

#if ACCELERATE_TENSOR_RT
std::ofstream ofstream("tensorrt-output.csv");
#elif ACCELERATE_FRUGALLY_DEEP
std::ofstream ofstream("frugallydeep-output.csv");
#elif ACCELERATE_ONNX_RUNTIME_CUDA
std::ofstream ofstream("onnxruntime-cuda-output.csv");
#else
std::ofstream ofstream("unknown-output.csv")
#endif

float test_parallel(const AbstractInference &inference, const std::vector<cv::Mat> &images) {
    const auto par_tensors = inference.predict_all(images, MODEL_OUTPUT_CLASS);

    ofstream << "Offset;Duration;Class;\n";
    for (const auto &par_tensor: par_tensors.out_tensors) {
        ofstream << par_tensor.offset_milliseconds << ";" << par_tensor.milliseconds << ";" << argmax(
            par_tensor.predictions) << ";\n";
    }
    return par_tensors.milliseconds;
}

float test_sequential(const AbstractInference &inference, const std::vector<cv::Mat> &images) {
    float t = 0;
    for (const auto &image: images) {
        const auto tensor = inference.predict(image, MODEL_OUTPUT_CLASS);
        t += tensor.milliseconds;
    }
    return t;
}

AbstractInference *create_inference(const std::string &model_path) {
#if ACCELERATE_TENSOR_RT
    return new TensorRTInference(model_path, nvinfer1::ILogger::Severity::kINFO);
#elif ACCELERATE_FRUGALLY_DEEP
    return new FrugallyDeepInference(model_path);
#elif ACCELERATE_ONNX_RUNTIME_CUDA
    return new OnnxRuntimeCudaInference(model_path);
#else
    throw std::runtime_error("Inference type not supported (Unknown).");
#endif
}

std::string to_out_string(const float f) {
    if (f == 0) {
        return "-";
    }
    return std::to_string(f);
}

void test(const std::string &model, const std::string &data) {
    const auto inference = create_inference(model);
    const auto images = load_images(data);
    constexpr auto total = 10;
    std::vector<float> parallel_run;
    std::vector<float> sequential_run;

    parallel_run.resize(total);
    sequential_run.reserve(total);
    for (int i = 0; i < total; i++) {
        //parallel_run.push_back(test_parallel(*inference, images));
        std::cout << "Test " << i + 1 << " / " << total << std::endl;
        sequential_run.push_back(test_sequential(*inference, images));
    }
    float sum_parallel = 0, sum_sequential = 0;
    for (int i = 0; i < total; i++) {
        std::cout << " & " << sequential_run[i] << " & " << to_out_string(parallel_run[i]) << "\\\\" <<
                std::endl;;
        sum_parallel += parallel_run[i];
        sum_sequential += sequential_run[i];
    }
    std::cout << "Průměr & " << sum_sequential / total << " & " << to_out_string(sum_parallel / total) <<
            "\\\\" << std::endl;
    delete inference;
}

int main(const int argc, char *argv[]) {
    const auto arguments = get_args(argc, argv);
    const auto model = arguments.at("model");
    const auto data = arguments.at("data_root");
    test(model, data);
}
