#include <inferencetools/FrugallyDeepEngineSequential.hpp>

std::unique_ptr<Tensor> FrugallyDeepEngineSequential::predict(const cv::Mat& predictionItem,
                                                              const std::function<Tensor(cv::Mat)>& transformer) const
{
    const auto tensor = transformer(predictionItem);
    const auto fdeepTensor = this->_toFdeepTensor(tensor, predictionItem);
    const auto outputTensors = this->model_->predict({fdeepTensor});

    const auto floatVector = outputTensors.at(0).to_vector();
    return std::make_unique<Tensor>(floatVector);
}

ImageSize FrugallyDeepEngineSequential::getSize(const cv::Mat& def) const
{
    return this->_getSize(def);
}

void FrugallyDeepEngineSequential::loadModel(const EngineInfo& engineInfo)
{
    this->_loadModel(engineInfo);
}

InferenceType FrugallyDeepEngineSequential::getType() const
{
    return this->engine_info_->type;
}
