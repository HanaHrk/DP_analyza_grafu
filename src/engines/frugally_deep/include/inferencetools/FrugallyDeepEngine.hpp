#pragma once
#include <fdeep/fdeep.hpp>
#include <inferencetools/InferenceEngine.hpp>

class FrugallyDeepEngine
{
protected:
    [[nodiscard]] ImageSize _getSize(const cv::Mat& def) const;

    void _loadModel(const EngineInfo& engineInfo);

    [[nodiscard]] fdeep::tensor _toFdeepTensor(const Tensor& tensor, const cv::Mat &def) const;

    std::unique_ptr<fdeep::model> model_;
    std::unique_ptr<EngineInfo> engine_info_;
};
