#pragma once

#include <inferencetools/InferenceEngine.hpp>
#if _MSC_VER
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#else
#include <onnxruntime_cxx_api.h>
#endif

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

class TensorRTEngineSequential final : public InferenceEngineSequential
{
public:
    ~TensorRTEngineSequential() override = default;

    void loadModel(const std::string& modelPath) override;

    [[nodiscard]] Tensor predict(const InferInput& input) const override;

private:
    std::unique_ptr<Ort::Env> env_{};
    std::unique_ptr<Ort::Session> session_{};
    Ort::MemoryInfo memoryInfo_{nullptr};
    std::unique_ptr<Ort::Allocator> allocator_;

    [[nodiscard]] TensorShape getInputTensorShape(const InferInput& input) const;

    [[nodiscard]] TensorShape getOutputTensorShape(const InferInput& input) const;

    [[nodiscard]] std::string getInputTensorName() const;

    [[nodiscard]] std::string getOutputTensorName() const;

    static uint64_t getTensorSize(const TensorShape& tensorShape);

    static Ort::SessionOptions createSessionOptions();

    static Ort::MemoryInfo createMemoryInfo();

    static Ort::ThreadingOptions createThreadingOptions();
};
