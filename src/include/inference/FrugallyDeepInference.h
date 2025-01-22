#pragma once
#if ACCELERATE_FRUGALLY_DEEP

#include <fdeep/fdeep.hpp>
#include <execution>
#include "../AbstractInference.h"


/**
 * @class ParallelStrategy
 *
 * @brief A class that defines and manages parallel execution strategies to
 *        optimize computational tasks across multiple processing units.
 *
 * This class provides mechanisms to distribute and balance workloads efficiently
 * in a parallelized environment. It enables the configuration of task parallelism
 * while abstracting the complexity of low-level threading or parallel library management.
 *
 * The class is designed to ensure scalability and performance for applications
 * requiring high-throughput computation or concurrent task execution, making it
 * suitable for multi-core or distributed system architectures.
 */
enum class ParallelStrategy: int
{
    STD_PARALLEL_FOREACH = 0,
    STD_THREADING = 1
};

/**
 * @class FrugallyDeepInference
 *
 * @brief A class designed to perform deep learning inference using models
 *        trained with Keras and converted to the frugally-deep format.
 *
 * This class encapsulates the functionality required to load, manage, and perform
 * forward prediction for deep learning models. It uses frugally-deep library as
 * the back-end for computational operations.
 *
 * The class enables initializing models from files, handling input data,
 * and generating the inference results efficiently by leveraging the frugally-deep
 * framework. It is ideal for scenarios where lightweight deep learning inference
 * is required without the overhead of a heavyweight deep learning library.
 */
class FrugallyDeepInference final : public AbstractInference
{
    /**
     * @class model_
     *
     * @brief Represents the core abstraction or structure for a specific
     *        computational or algorithmic model within the application.
     *
     * This class or structure serves as a foundational element to encapsulate
     * data, behaviors, and interactions needed to define a comprehensive model.
     * It enables the representation, manipulation, and evaluation of the model
     * using defined interfaces and methods, adhering to the required design principles.
     *
     * Designed to be extensible and flexible, it aims to facilitate ease of use
     * and integration within larger systems or workflows, supporting modularity
     * and maintainability.*/
    fdeep::model model_;

    /**
     * @var parallel_strategy_
     *
     * @brief A variable that holds the current configuration or instance of the parallel execution strategy.
     *
     * This variable is utilized to manage and apply the selected strategy for parallelizing
     * computational workloads. It facilitates efficient task distribution and synchronization
     * tailored to the underlying hardware and application requirements.
     *
     * The purpose of this variable is to ensure the execution framework adheres to the
     * predefined parallel strategy, optimizing resource utilization and performance in
     * multi-threaded or distributed processing environments.
     */
    ParallelStrategy parallel_strategy_;

    /**
     * @brief Converts an OpenCV image (cv::Mat) into a frugally-deep tensor
     *        compatible with the model's input format.
     *
     * This method processes an OpenCV image by resizing and transforming it into
     * a tensor representation suitable for inference with the frugally-deep library.
     * It ensures the output matches the input shape and depth requirements of the loaded
     * model.
     *
     * @param image The input image as an OpenCV cv::Mat object to be transformed into a tensor.
     * @return A frugally-deep tensor containing the transformed image data.
     */
    [[nodiscard]] fdeep::tensor to_tensor(const cv::Mat& image) const;

    /**
     * @brief Provides a callback function for performing inference using the FrugallyDeep model.
     *
     * This method returns a callable function that executes inference on the provided input data.
     * The callback manages time measurements, performs predictions using the FrugallyDeep model,
     * and updates the output tensors with the results, including elapsed times and predictions.
     *
     * It is designed to be used in multi-threaded environments, where the atomic counter (`index`)
     * ensures thread-safe updates to the output tensor vector.
     *
     * @return A function object that performs inference on an input image, computes elapsed times,
     *         and updates the provided output tensor vector.
     */
    [[nodiscard]] std::function<void(const std::chrono::time_point<std::chrono::steady_clock>,
                                     std::vector<OutParTensor>&, std::atomic<int>&,
                                     const cv::Mat&)> inference_callback() const;

    /**
     * @brief Executes inference in parallel on a batch of input images and stores the results in output tensors.
     *
     * This method utilizes different parallel execution strategies (e.g., parallel for-each or threading)
     * based on the configured parallelization strategy. It ensures efficient parallel processing of multiple
     * images and updates the respective output tensors concurrently. The behavior is tailored to leverage
     * available system resources for optimal performance.
     *
     * @param images A vector of input images (cv::Mat) on which inference needs to be executed.
     * @param out_parallel_tensors A vector of output tensors where the inference results will be stored.
     * @param index An atomic integer that acts as a shared counter to handle indexing or task tracking.
     */
    void run_parallel(const std::vector<cv::Mat>& images,
                      std::vector<OutParTensor>& out_parallel_tensors,
                      std::atomic<int>& index) const;

public:
    explicit FrugallyDeepInference(const std::string& model_path);

    ~FrugallyDeepInference() override;

    /**
     * @brief Sets the parallel execution strategy for the inference process.
     *
     * This method configures the parallel strategy used by the inference system
     * to optimize workload distribution and enhance computational performance
     * across processing units.
     *
     * @param parallel_strategy The parallel execution strategy to be applied.
     */
    void set_parallel_strategy(const ParallelStrategy& parallel_strategy);

    /**
     * @brief A function or method responsible for generating predictions based
     *        on the provided input data using a pre-trained model.
     *
     * The predict function utilizes a trained model to infer results or classify
     * data. Depending on its implementation, it may support various input formats,
     * output types, and optional parameters to customize predictions.
     *
     * This function is instrumental in applying machine learning or statistical
     * models in real-world scenarios, making it a key component in analytics
     * pipelines and intelligent systems.
     *
     * @param image The input data based on which predictions are generated.
     *                  This can be a single instance, a batch of instances, or any
     *                  other data structure compatible with the model.
     * @return The predicted outcome or results, which could be a scalar, a vector,
     *         a matrix, or any other structure, depending on the model and input.
     */
    [[nodiscard]] OutTensor predict(const cv::Mat& image) const override;

    /**
     * @function predict_all
     *
     * @brief Computes predictions for all data points provided as input, using the
     *        trained model.
     *
     * This method processes the given input data in bulk and generates corresponding
     * predictions based on the model's parameters. It is optimized for batch
     * processing and ensures accurate results across all given instances. The method
     * requires the input data to be formatted as expected by the model.
     *
     * @param images A collection of input instances for which predictions are to be generated.
     *             The format and type of this parameter depend on the specific model's requirements.
     * @return A collection of predictions corresponding to the input data. The structure
     *         of the returned predictions matches the design of the model's output.
     */
    [[nodiscard]] OutParTensors predict_all(const std::vector<cv::Mat>& images) const override;
};
#endif
