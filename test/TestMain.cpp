#include <ImageUtils.h>
#include <fstream>
#include "AbstractInference.h"
#include "inference/FrugallyDeepInference.h"
#include "inference/TensorRTInference.h"
#include "TestUtils.h"
#include "MathUtils.h"
#include "inference/OnnxRuntimeCudaInference.h"

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

/**
 * @brief Converts a float value to a string representation for output purposes.
 *
 * This function transforms a given float value into a string format.
 * If the value is 0, it returns a placeholder string "-". For all other
 * cases, it converts the float to its string representation.
 *
 * @param f A float value to be converted into a string.
 *          If the value is 0, a specific placeholder is returned.
 * @return A string representing the given float value, or "-" if the input is 0.
 */
std::string to_out_string(const float f)
{
    if (f == 0)
    {
        return "-";
    }
    return std::to_string(f);
}

/**
 * @brief Executes parallel inference on a set of input images and records the results.
 *
 * This function processes a batch of images in parallel by invoking the inference model's `predict_all` method.
 * It writes the inference results, including offset time, duration, and predicted class, to an output file.
 * The total duration of all parallel inferences is then returned.
 *
 * @param inference The inference model implementing the AbstractInference interface.
 * @param images A vector of OpenCV Mat objects representing the input images to be processed.
 * @return The total inference duration in milliseconds for all the images processed in parallel.
 */
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

/**
 * @brief Executes sequential inference on a set of input images and calculates the total duration.
 *
 * This function processes a vector of images one by one, invoking the inference model for each image.
 * It accumulates the inference duration for each image into a total duration and returns it.
 *
 * @param inference The inference model implementing the AbstractInference interface.
 * @param images A vector of OpenCV Mat objects representing the input images to be processed.
 * @return The total inference duration in milliseconds for all the images.
 */
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

/**
 * @brief Creates an inference object based on the specified model and available acceleration backend.
 *
 * This function determines the type of inference backend to use depending on the compilation flags
 * (e.g., TensorRT, FrugallyDeep, or ONNX Runtime with CUDA). It then instantiates and returns the
 * corresponding inference object. If no supported backend is defined, it throws an exception.
 *
 * @param model_path The file path to the trained inference model.
 * @return A pointer to an AbstractInference object configured for the specified model.
 * @throws std::runtime_error If no supported inference backend is available.
 */
AbstractInference* create_inference(const std::string& model_path)
{
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


/**
 * @brief Executes a single test iteration on both sequential and parallel inference modes.
 *
 * This function measures the performance of sequential and parallel inference for a given
 * test iteration. It executes the inference using the provided images and tracks
 * the execution times for both sequential and parallel runs. The results of these runs
 * are appended to the respective vectors for further analysis and are displayed after
 * the execution.
 *
 * @param iteration The current test iteration number.
 * @param sequential_run A vector storing the execution times of all completed sequential tests.
 * @param parallel_run A vector storing the execution times of all completed parallel tests.
 * @param inference A pointer to an AbstractInference object representing the inference model to be tested.
 * @param images A vector of input images to be used in the test.
 */
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


/**
 * @brief Executes a series of tests comparing sequential and parallel inference times.
 *
 * This function initializes an inference engine with the given model, loads images
 * from the data path, and runs a predetermined number of tests. Each test measures
 * both sequential and parallel inference performance. Results are processed and displayed
 * after each iteration.
 *
 * @param model_path Path to the model file used for inference.
 * @param data_path Path to the directory containing image data for testing.
 * @return Returns 0 upon successful completion of all tests.
 */
int evaluate_model_performance(const std::string& model_path, const std::string& data_path)
{
    const auto inference = create_inference(model_path);
    const auto images = load_images(data_path);

    constexpr int total_tests = 10;
    std::vector<float> parallel_run, sequential_run;

#if ACCELERATE_FRUGALLY_DEEP
    reinterpret_cast<FrugallyDeepInference*>(inference)->set_parallel_strategy(ParallelStrategy::STD_PARALLEL_FOREACH);
#endif
    for (int i = 0; i < total_tests; i++)
    {
        evaluate_inference_performance(i, sequential_run, parallel_run, inference, images);
    }
#if ACCELERATE_FRUGALLY_DEEP
    reinterpret_cast<FrugallyDeepInference*>(inference)->set_parallel_strategy(ParallelStrategy::STD_THREADING);
    for (int i = 0; i < total_tests; i++)
    {
        evaluate_inference_performance(i, sequential_run, parallel_run, inference, images);
    }
#endif
    delete inference;
    return 0;
}

/**
 * @brief Displays usage instructions and available options for the application.
 *
 * This function outputs to the console the required command-line arguments for
 * executing the application. It specifies the format for providing the inference
 * model path, data path, and an optional help flag. The function serves as a
 * guide for users by describing the purpose of each supported command-line
 * argument.
 */
void print_usage()
{
    std::cout << "Usage: InferenceTest --model <model_path> --data <data_path>" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --model    Path to the inference model file (required)" << std::endl;
    std::cout << "  --data     Path to the data directory for testing (required)" << std::endl;
    std::cout << "  --help     Display this help message" << std::endl;
}

/**
 * @brief Processes command-line arguments and executes the test function.
 *
 * This function validates the presence of the required command-line arguments
 * `--model` and `--data`. If either of these arguments is missing, it prints a
 * usage guide to the console via `print_usage()` and exits with a return code of 1.
 * Otherwise, it retrieves the paths for the inference model and data input
 * and passes them to the `test` function for execution.
 *
 * @param args A map containing the parsed command-line arguments where keys
 *             are argument names and values are their corresponding values.
 *
 * @return int Returns 1 if required arguments are missing, otherwise returns
 *             the result of the `test` execution.
 */
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

/**
 * @brief Main execution point for the program.
 *
 * This function initializes the command-line argument parser, retrieves the
 * parsed arguments, and processes them. It delegates further execution by
 * calling the `process_args` function, which handles the main logic based
 * on the parsed arguments.
 *
 * @param argc The number of command-line arguments, including the program name.
 * @param argv Array of C-style strings containing the command-line arguments.
 *             These arguments are expected to include options for specifying
 *             the inference model and data paths.
 *
 * @return int Returns the exit code of the program. This value is obtained
 *             directly from the result of the `process_args` function.
 */
int main(const int argc, char* argv[])
{
    const auto arguments = get_args(argc, argv);
    return execute_model_test(arguments);
}
