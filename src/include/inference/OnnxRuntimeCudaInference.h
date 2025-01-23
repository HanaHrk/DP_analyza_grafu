#pragma once

#if ACCELERATE_ONNX_RUNTIME_CUDA
#include "AbstractInference.h"
#include <onnxruntime_cxx_api.h>
#include <semaphore>
#include <cuda_runtime.h>

enum class ExecutionProvider
{
    DEFAULT,
    CUDA,
    OPEN_VINO_GPU,
    OPEN_VINO_CPU,
};

/**
 * @class CudaMemoryDeleter
 * @brief A custom deleter for CUDA-allocated device memory.
 *
 * This class provides a callable functional object which is used to release
 * memory that was allocated on the CUDA device. It ensures that the memory
 * is properly freed using cudaFree, preventing memory leaks and ensuring
 * safe management of GPU resources.
 *
 * Useful in combination with smart pointers like std::unique_ptr or
 * std::shared_ptr for managing device memory automatically.
 *
 * The class is typically invoked when a smart pointer managing CUDA device
 * memory goes out of scope, thus automatically freeing the corresponding
 * GPU memory.
 */
struct CudaMemoryDeleter
{
    /**
     * @class CudaMemoryDeleter
     * @brief A custom deleter for managing CUDA device memory lifecycle.
     *
     * This class serves as a functional object to handle de-allocation
     * of memory allocated on the CUDA device, ensuring proper resource cleanup.
     */
    explicit CudaMemoryDeleter(Ort::Allocator* alloc)
    {
        alloc_ = alloc;
    }

    void operator()(void* ptr) const
    {
        this->alloc_->Free(ptr);
    }

    Ort::Allocator* alloc_;
};

class OnnxRuntimeCudaInference final : public AbstractInference
{
    std::unique_ptr<Ort::Env> env_{};
    std::unique_ptr<Ort::Session> session_{};
    std::unique_ptr<Ort::MemoryInfo> memory_info_;
    std::unique_ptr<Ort::Allocator> cuda_allocator_;

    static Ort::SessionOptions build_session_options(ExecutionProvider execution_provider);

    static void async_callback(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr status);

    [[nodiscard]] std::string get_input_tensor_name() const;

    [[nodiscard]] std::string get_output_tensor_name() const;

    static size_t calculate_size_from_shape(const std::vector<int64_t>& shape);

    [[nodiscard]] std::pair<std::vector<int64_t>, std::vector<int64_t>> get_input_output_shapes() const;

    static std::pair<cudaEvent_t, cudaEvent_t> OnnxRuntimeCudaInference::initializeCudaEvents();


    static std::vector<float> process_image(const cv::Mat& image, int width, int height);

public:
    explicit OnnxRuntimeCudaInference(const std::string& model_path,
                                      ExecutionProvider provider = ExecutionProvider::CUDA);


    ~OnnxRuntimeCudaInference() override;

    /**
     * @brief Performs inference on the input image using the ONNX Runtime with CUDA.
     *
     * This method processes the given input image, performs necessary preprocessing,
     * allocates required GPU memory, runs the inference on the CUDA-enabled ONNX runtime,
     * and retrieves the output predictions. It also measures and returns the inference time.
     *
     * The method assumes that the ONNX model is already loaded into the session
     * and the necessary configurations (input/output tensor names, memory allocations)
     * have been set up.
     *
     * @param image The input image to be processed and used for inference. It must be in a format
     *              compatible with the model's input requirements.
     * @return An OutTensor structure containing predictions (output class probabilities) and
     *         the elapsed inference time in milliseconds.
     */
    [[nodiscard]] OutTensor predict(const cv::Mat& image) const override;

    [[nodiscard]] OutParTensors predict_all(const std::vector<cv::Mat>& images) const override;
};

#endif
