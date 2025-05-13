#pragma once
#include <fdeep/fdeep.hpp>
#include <inferencetools/InferenceEngine.hpp>


class FrugallyDeepEngine final : public InferenceEngineSequential, public InferenceEngineParallel
{
public:
    enum class ParallelMode
    {
        STD_THREADING,
        PARALLEL_FOREACH
    };

public:
    void loadModel(const std::string& enginePath) override;

    Tensor predict(const InferInput& input) const override;

    std::vector<Tensor> predictAll(const std::vector<InferInput>& input) const override;

    std::vector<Tensor> predictAll(const std::vector<InferInput>& inputs, const ParallelMode& mode) const;

private:
    std::unique_ptr<fdeep::model> model_;

    [[nodiscard]] fdeep::tensor toFdeepTensor(const InferInput& input) const;

    ImageShape modelInputShape() const;

    std::function<void(const InferInput& input, std::vector<Tensor>& output, std::atomic_int& index)> inferenceCallback() const;
};
