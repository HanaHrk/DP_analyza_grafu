#pragma once

#if ACCELERATE_TENSOR_RT
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "AbstractInference.h"
#include "Logger.h"

class TensorRTInference final : public AbstractInference
{
    static constexpr int input_index = 0;
    static constexpr int output_index = 1;

    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<ProgramLogger> program_logger_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::array<void*, 2> tensor_buffers_{};

    static std::vector<float> process_image(const cv::Mat& image, int width, int height);

    static float compute_milliseconds(const cudaEvent_t& start, const cudaEvent_t& end);

    [[nodiscard]] std::string get_input_tensor_name() const;

    [[nodiscard]] std::string get_output_tensor_name() const;

    [[nodiscard]] size_t get_tensor_size(const std::string& tensor_name) const;

    static void create_and_record_event(cudaEvent_t& event, const cudaStream_t& stream);

    [[nodiscard]] OutParTensor process_image_inference(const cudaStream_t& stream,
                                                       const cudaEvent_t& start_event,
                                                       const std::vector<float>& input_data,
                                                       std::size_t input_size,
                                                       std::size_t output_size) const;

    void init_tensor_buffers();

    static nvinfer1::ICudaEngine* build_engine(const std::string& onnx_model_path, ProgramLogger& logger);

public:
    explicit TensorRTInference(const std::string& model_path, const nvinfer1::ILogger::Severity& severity);

    ~TensorRTInference() override;

    /**
     * Runs inference on the provided input image using TensorRT and returns the prediction results.
     *
     * @param image The input image in cv::Mat format. It should be preprocessed to match the model's
     *              input requirements, such as dimensions and data format.
     * @return An OutTensor struct containing the prediction results. The `predictions` field includes
     *         probabilities or scores for output classes, and the `milliseconds` field indicates the
     *         inference time in milliseconds.
     */
    [[nodiscard]] OutTensor predict(const cv::Mat& image) const override;

    /**
     * Performs batch inference on multiple input images using TensorRT and returns the prediction results
     * for all images, along with the cumulative inference time.
     *
     * @param images A vector of input images in cv::Mat format. Each image should be preprocessed
     *               to match the model's input requirements, including dimensions and data format.
     * @return An OutParTensors struct containing the prediction results for all input images. The `out_tensors`
     *         field includes a vector of individual predictions, while the `milliseconds` field indicates
     *         the total elapsed time for processing all images in milliseconds.
     */
    [[nodiscard]] OutParTensors predict_all(const std::vector<cv::Mat>& images) const override;
};

#endif
