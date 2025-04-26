#pragma once
#include <fdeep/fdeep.hpp>
#include <inferencetools/InferenceEngine.hpp>

class FrugallyDeepEngine
{
protected:
    void _loadModel(const std::string& enginePath);

    [[nodiscard]] fdeep::tensor _toFdeepTensor(const InferInput& input) const;

    std::unique_ptr<fdeep::model> model_;

private:
    ImageShape _modelInputShape() const;
};
