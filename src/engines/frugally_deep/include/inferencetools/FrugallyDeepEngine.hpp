#pragma once
#include <fdeep/fdeep.hpp>
#include <inferencetools/InferenceEngine.hpp>

class FrugallyDeepEngine final : public InferenceEngine
{
    std::unique_ptr<fdeep::model> model_;

public:
    FrugallyDeepEngine() = default;

    std::unique_ptr<ClassificationResult> classify(const Tensor& tensor) override;

    std::unique_ptr<SegmentationResult> segment(const Tensor& tensor) override;

    void loadModel(const std::string& modelPath) override;

    ~FrugallyDeepEngine() override;

    [[nodiscard]] ImageSize getSize() const override;

private:
    [[nodiscard]] fdeep::tensor toFdeepTensor(const Tensor& tensor) const;
};
