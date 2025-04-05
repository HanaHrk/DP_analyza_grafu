#pragma once
#include <inferencetools/InferenceEngine.hpp>
#include <inferencetools/FrugallyDeepEngine.hpp>

class FrugallyDeepEngineSequential final : protected FrugallyDeepEngine, public InferenceEngineSequential
{
public:
    FrugallyDeepEngineSequential() = default;

    std::unique_ptr<Tensor> predict(const cv::Mat& predictionItem,
                                    const std::function<Tensor(cv::Mat)>& transformer) const override;

    [[nodiscard]] ImageSize getSize(const cv::Mat& def) const override;

    void loadModel(const EngineInfo& engineInfo) override;

    [[nodiscard]] InferenceType getType() const override;
};
