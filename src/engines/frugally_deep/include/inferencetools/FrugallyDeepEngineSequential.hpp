#pragma once
#include <inferencetools/InferenceEngine.hpp>
#include <inferencetools/FrugallyDeepEngine.hpp>

class FrugallyDeepEngineSequential final : protected FrugallyDeepEngine, public InferenceEngineSequential
{
public:
    FrugallyDeepEngineSequential() = default;

    void loadModel(const std::string& enginePath) override;

    Tensor predict(const InferInput& input) const override;

};
