#include <inferencetools/FrugallyDeepEngine.hpp>

FrugallyDeepEngine::~FrugallyDeepEngine() = default;

std::unique_ptr<ClassificationResult> FrugallyDeepEngine::classify(const Tensor& tensor)
{
    const auto fdeepTensor = toFdeepTensor(tensor);
    const auto outputTensors = this->model_->predict({fdeepTensor});

    const auto confidences = outputTensors.at(0).to_vector();
    return std::make_unique<ClassificationResult>(getClass(confidences), "", confidences);
}

std::unique_ptr<SegmentationResult> FrugallyDeepEngine::segment(const Tensor& tensor)
{
    return nullptr;
}

void FrugallyDeepEngine::loadModel(const std::string& modelPath)
{
    const auto model = fdeep::load_model(modelPath);
    this->model_ = std::make_unique<fdeep::model>(model);
}


fdeep::tensor FrugallyDeepEngine::toFdeepTensor(const Tensor& tensor) const
{
    const auto imageSize = getSize();
    const auto shape = fdeep::tensor_shape(imageSize.height, imageSize.width, imageSize.depth);
    const auto fdeepTensor = fdeep::tensor{shape, tensor};
    return fdeep::tensor{shape, tensor};
}

ImageSize FrugallyDeepEngine::getSize() const
{
    const auto input_shapes = this->model_->get_input_shapes();
    const auto& input_shape = input_shapes.at(0);
    const auto height = input_shape.height_.get_with_default(0);
    const auto width = input_shape.width_.get_with_default(0);
    const auto depth = input_shape.depth_.get_with_default(0);
    return ImageSize{static_cast<int>(width), static_cast<int>(height), static_cast<int>(depth)};
}
