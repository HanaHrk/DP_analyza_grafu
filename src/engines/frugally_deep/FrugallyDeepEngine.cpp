#include <inferencetools/FrugallyDeepEngine.hpp>

#include "../../../benchmarks/include/ImageFormatException.h"

ImageSize FrugallyDeepEngine::_getSize(const cv::Mat& def) const
{
    const auto input_shapes = this->model_->get_input_shapes();
    const auto& input_shape = input_shapes.at(0);
    const auto height = input_shape.height_.get_with_default(0);
    const auto width = input_shape.width_.get_with_default(0);
    const auto depth = input_shape.depth_.get_with_default(0);
    if (height == 0 || width == 0 || depth == 0)
    {
        if (def.cols == 0 || def.rows == 0 || def.channels() == 0)
        {
            throw ImageFormatException("Could not determine image size.");
        }
        return ImageSize{def.cols, def.rows, def.channels()};
    }
    return ImageSize{static_cast<int>(width), static_cast<int>(height), static_cast<int>(depth)};
}

void FrugallyDeepEngine::_loadModel(const EngineInfo& engineInfo)
{
    const auto model = fdeep::load_model(engineInfo.path);
    this->model_ = std::make_unique<fdeep::model>(model);
    this->engine_info_ = std::make_unique<EngineInfo>(engineInfo);
}


fdeep::tensor FrugallyDeepEngine::_toFdeepTensor(const Tensor& tensor, const cv::Mat& def) const
{
    const auto imageSize = this->_getSize(def);
    const auto shape = fdeep::tensor_shape(imageSize.height, imageSize.width, imageSize.depth);
    const auto fdeepTensor = fdeep::tensor{shape, tensor};
    return fdeep::tensor{shape, tensor};
}
