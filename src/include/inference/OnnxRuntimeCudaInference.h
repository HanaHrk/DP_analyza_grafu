#pragma once

#if ACCELERATE_ONNX_RUNTIME_CUDA
#include "AbstractInference.h"
#include <onnxruntime_cxx_api.h>
#include <semaphore>

struct CudaMemoryDeleter {
    explicit CudaMemoryDeleter(Ort::Allocator *alloc) {
        alloc_ = alloc;
    }

    void operator()(void *ptr) const {
        this->alloc_->Free(ptr);
    }

    Ort::Allocator *alloc_;
};

class OnnxRuntimeCudaInference final : public AbstractInference {
    Ort::Env *env_{};
    Ort::Session *session_{};
    Ort::MemoryInfo *memory_info_;
    Ort::Allocator *cuda_allocator_;

    static Ort::SessionOptions build_session_options();

    static void async_callback(void *user_data, OrtValue **outputs, size_t num_outputs, OrtStatusPtr status);

public:
    explicit OnnxRuntimeCudaInference(const std::string &model_path);


    ~OnnxRuntimeCudaInference() override;

    [[nodiscard]] OutTensor predict(const cv::Mat &image, std::size_t out_class) const override;

    [[nodiscard]] OutParTensors predict_all(const std::vector<cv::Mat> &images, std::size_t out_class) const override;
};

#endif
