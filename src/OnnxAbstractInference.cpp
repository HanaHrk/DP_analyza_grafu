#if defined(ACCELERATE_ONNX_RUNTIME_CUDA) OR defined(ACCELERATE_ONNX_RUNTIME_OPEN_VINO_CPU) OR defined(ACCELERATE_ONNX_RUNTIME_OPEN_VINO_GPU)

#include "OnnxAbstractInference.h"

#include <numeric>

std::string OnnxAbstractInference::get_output_tensor_name() const
{
    const Ort::AllocatorWithDefaultOptions allocator;
    const auto output_name_prt = this->session_->GetOutputNameAllocated(0, allocator);
    return output_name_prt.get();
}

std::string OnnxAbstractInference::get_input_tensor_name() const
{
    const Ort::AllocatorWithDefaultOptions allocator;
    const auto input_name_prt = this->session_->GetInputNameAllocated(0, allocator);
    return input_name_prt.get();
}

size_t OnnxAbstractInference::calculate_total_size(const std::vector<int64_t>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>{}) * sizeof(float);
}

std::pair<std::vector<int64_t>, std::vector<int64_t>> OnnxAbstractInference::fetch_shapes() const
{
    const auto input_shape = this->session_->GetInputTypeInfo(0)
                                 .GetTensorTypeAndShapeInfo()
                                 .GetShape();
    const auto output_shape = this->session_->GetOutputTypeInfo(0)
                                  .GetTensorTypeAndShapeInfo()
                                  .GetShape();
    return {input_shape, output_shape};
}

Ort::Value OnnxAbstractInference::create_tensor(const Ort::MemoryInfo& memory_info, void* data,
                                                const size_t num_elements,
                                                const std::vector<int64_t>& shape)
{
    return Ort::Value::CreateTensor<float>(memory_info, static_cast<float*>(data), num_elements, shape.data(),
                                           shape.size());
}
#endif
