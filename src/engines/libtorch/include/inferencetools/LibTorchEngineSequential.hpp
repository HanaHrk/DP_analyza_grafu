#pragma once
#include <torch/script.h>
#include <inferencetools/InferenceEngine.hpp>


class LibtorchEngineSequential final : public InferenceEngineSequential
{
public:
    ~LibtorchEngineSequential() override = default;

    void loadModel(const std::string& enginePath) override;

    [[nodiscard]] Tensor predict(const InferInput& input) const override;

private:
    std::unique_ptr<torch::jit::script::Module> model_;
};
