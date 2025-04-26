#include <inferencetools/FrugallyDeepEngineSequential.hpp>


Tensor FrugallyDeepEngineSequential::predict(const InferInput& input) const
{
    const auto fdeepTensor = this->_toFdeepTensor(input);
    const auto outputTensors = this->model_->predict({fdeepTensor});
    const auto floatVector = outputTensors.at(0).to_vector();
    return Tensor{floatVector};
}


void FrugallyDeepEngineSequential::loadModel(const std::string& enginePath)
{
    this->_loadModel(enginePath);
}
