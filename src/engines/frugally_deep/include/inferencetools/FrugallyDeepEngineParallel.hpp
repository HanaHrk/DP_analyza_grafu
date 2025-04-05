#pragma once
#include <fdeep/fdeep.hpp>
#include <inferencetools/InferenceEngine.hpp>

class FrugallyDeepEngineParallel final : public InferenceEngineParallel
{
public:
     ~FrugallyDeepEngineParallel() override = default;

    std::vector<std::unique_ptr<Tensor>> segmentAll(const std::vector<Tensor>& images) override;

    std::vector<std::unique_ptr<Tensor>> classifyAll(const std::vector<Tensor>& images) override;

};
