#include <inferencetools/FrugallyDeepEngine.hpp>

#include <inferencetools/exception/EngineRuntimeException.hpp>


void FrugallyDeepEngine::_loadModel(const std::string& enginePath)
{
    const auto model = fdeep::load_model(enginePath);
    this->model_ = std::make_unique<fdeep::model>(model);
}

ImageShape FrugallyDeepEngine::_modelInputShape() const
{
    const auto inputShapes = this->model_->get_input_shapes();
    const auto& inputShape = inputShapes.at(0);
    const auto height = inputShape.height_.get_with_default(0);
    const auto width = inputShape.width_.get_with_default(0);
    const auto depth = inputShape.depth_.get_with_default(0);
    return ImageShape{height, width, depth};
}


fdeep::tensor FrugallyDeepEngine::_toFdeepTensor(const InferInput& input) const
{
    auto inputWidth = input.inputWidth;
    auto inputHeight = input.inputHeight;
    auto inputDepth = input.inputDepth;
    const auto modelSize = this->_modelInputShape();

    if (inputWidth == MODEL_PROPERTIES)
    {
        inputWidth = modelSize.width;
    }
    if (inputHeight == MODEL_PROPERTIES)
    {
        inputHeight = modelSize.height;
    }
    if (inputDepth == MODEL_PROPERTIES)
    {
        inputDepth = modelSize.depth;
    }
    if (inputWidth == UNKNOWN_PROPERTY || inputHeight == UNKNOWN_PROPERTY || inputDepth == UNKNOWN_PROPERTY)
    {
        throw EngineRuntimeException(
            "Input Shape is not defined properly. Model or user must define input shape for inference.");
    }
    const auto shape = fdeep::tensor_shape(inputWidth, inputHeight, inputDepth);
    return fdeep::tensor{shape, input.input};
}
