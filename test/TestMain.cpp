#include <fstream>
#include "AbstractInference.h"
#include "FrugallyDeepInference.h"
#include "TensorRTInference.h"
#include "TestUtils.h"
#include "MathUtils.h"
#include "OnnxCudaInference.h"
#include "OnnxOpenVinoCpuInference.h"

const std::string TENSORRT_OUTPUT_FILE = "tensorrt-output.csv";
const std::string FRUGALLY_DEEP_OUTPUT_FILE = "frugallydeep-output.csv";
const std::string ONNX_RUNTIME_CUDA_OUTPUT_FILE = "onnxruntime-cuda-output.csv";
const std::string UNKNOWN_OUTPUT_FILE = "unknown-output.csv";


const std::string MODEL_ARG = "model";
const std::string DATA_ARG = "data";
const std::string HELP_ARG = "help";

#if ACCELERATE_TENSOR_RT
std::ofstream ofstream(TENSORRT_OUTPUT_FILE);
#elif ACCELERATE_FRUGALLY_DEEP
std::ofstream ofstream(FRUGALLY_DEEP_OUTPUT_FILE);
#elif ACCELERATE_ONNX_RUNTIME_CUDA
std::ofstream ofstream(ONNX_RUNTIME_CUDA_OUTPUT_FILE);
#else
std::ofstream ofstream(UNKNOWN_OUTPUT_FILE);
#endif


std::string to_out_string(const float f)
{
    if (f == 0)
    {
        return "-";
    }
    return std::to_string(f);
}

float test_parallel(const AbstractInference& inference, const std::vector<cv::Mat>& images)
{
    const auto [out_tensors, milliseconds] = inference.predict_all(images);
    ofstream << "Offset;Duration;Class;\n";
    for (const auto& [predictions, milliseconds, offset_milliseconds] : out_tensors)
    {
        ofstream << offset_milliseconds << ";" << milliseconds << ";"
            << argmax(predictions) << ";\n";
    }
    return milliseconds;
}

float test_sequential(const AbstractInference& inference, const std::vector<cv::Mat>& images)
{
    float total_duration = 0;
    for (const auto& image : images)
    {
        const auto [predictions, milliseconds] = inference.predict(image);
        total_duration += milliseconds;
    }
    return total_duration;
}


std::vector<std::pair<std::string, std::unique_ptr<AbstractInference>>> create_inference(const std::string& model_path)
{
    std::vector<std::pair<std::string, std::unique_ptr<AbstractInference>>> inference_engines;
#if defined(ACCELERATE_TENSOR_RT)
    inference_engines.emplace_back("TensorRT", std::make_unique<TensorRTInference>(model_path));
#endif

#if defined(ACCELERATE_FRUGALLY_DEEP)
    inference_engines.emplace_back("Frugally Deep", std::make_unique<FrugallyDeepInference>(model_path));
#endif

#if defined(ACCELERATE_ONNX_RUNTIME_CUDA)
    inference_engines.emplace_back("Onnx Runtime CUDA", std::make_unique<OnnxCudaInference>(model_path));
#endif

#if defined(ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU)
    inference_engines.emplace_back("Onnx Runtime OpenVINO CPU", std::make_unique<OnnxOpenVinoCpuInference>(model_path));
#endif

#if defined(ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU)
    inference_engines.emplace_back("Onnx Runtime OpenVINO GPU", std::make_unique<OnnxOpenVinoCpuInference>(model_path));
#endif

    return inference_engines;
}

void evaluate_inference_performance(const int iteration, std::vector<float>& sequential_run,
                                    std::vector<float>& parallel_run,
                                    const AbstractInference* inference, const std::vector<cv::Mat>& images)
{
    std::cout << "Iteration: " << iteration << std::endl;
    sequential_run.push_back(test_sequential(*inference, images));
    std::cout << "Sequential time: ";
    std::cout << sequential_run[iteration];
    std::cout << " ms" << std::endl;
    parallel_run.push_back(test_parallel(*inference, images));
    std::cout << "Parallel time: ";
    std::cout << parallel_run[iteration];
    std::cout << " ms" << std::endl;
}

int evaluate_model_performance(const std::string& model_path, const std::string& data_path)
{
    const auto inference = create_inference(model_path);
    const auto images = load_images(data_path);
    for (const auto& [name, inference] : inference)
    {
        constexpr int iterations = 5;
        std::vector<float> sequential_run;
        std::vector<float> parallel_run;
        std::cout << "Running Inference Type: " << name << std::endl;
        for (int i = 0; i < iterations; ++i)
        {
            evaluate_inference_performance(i, sequential_run, parallel_run, inference.get(), images);
        }
    }
    return 0;
}

void print_usage()
{
    std::cout << "Usage: InferenceTest --model <model_path> --data <data_path>" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --model    Path to the inference model file (required)" << std::endl;
    std::cout << "  --data     Path to the data directory for testing (required)" << std::endl;
    std::cout << "  --help     Display this help message" << std::endl;
}

int execute_model_test(const std::map<std::string, std::string>& args)
{
    if (has_args(args, {HELP_ARG}))
    {
        print_usage();
        return 0;
    }
    if (!has_args(args, {MODEL_ARG, DATA_ARG}))
    {
        print_usage();
        return 1;
    }
    const auto& model_path = args.at(MODEL_ARG);
    const auto& data_path = args.at(DATA_ARG);
    return evaluate_model_performance(model_path, data_path);
}

int main(const int argc, char* argv[])
{
    const auto arguments = get_args(argc, argv);
    return execute_model_test(arguments);
}
