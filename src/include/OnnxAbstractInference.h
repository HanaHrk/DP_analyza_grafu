#pragma once

#if defined ACCELERATE_ONNX_RUNTIME_CUDA || defined ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU || defined ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU

#include <onnxruntime_cxx_api.h>

#include "AbstractInference.h"

struct MemoryDeleter
{
    explicit MemoryDeleter(Ort::Allocator* alloc)
    {
        alloc_ = alloc;
    }

    void operator()(void* ptr) const
    {
        this->alloc_->Free(ptr);
    }

    Ort::Allocator* alloc_;
};

class OnnxAbstractInference : public AbstractInference
{
protected:
    std::unique_ptr<Ort::Env> env_{};
    std::unique_ptr<Ort::Session> session_{};
    Ort::MemoryInfo memory_info_{nullptr};
    std::unique_ptr<Ort::Allocator> allocator_;

    [[nodiscard]] std::string get_output_tensor_name() const;

    [[nodiscard]] std::string get_input_tensor_name() const;

    static size_t calculate_total_size(const std::vector<int64_t>& shape);

    [[nodiscard]] std::pair<std::vector<int64_t>, std::vector<int64_t>> fetch_shapes() const;

    static Ort::Value create_tensor(const Ort::MemoryInfo& memory_info, void* data, size_t num_elements, const std::vector<int64_t>& shape);
public:
    virtual Ort::SessionOptions build_session_options() = 0;

    virtual Ort::MemoryInfo build_memory_info() = 0;
};
#endif
